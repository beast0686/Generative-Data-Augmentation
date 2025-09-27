from abc import ABC
from pathlib import Path

import torch
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.logger import get_logger

logger = get_logger(__name__)


class Result(BaseModel):
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    classification_report: dict
    confusion_matrix: list[list[int]]


class Runner(ABC):
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def _train_one(
        self,
        model: Module,
        loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: Module,
        epoch_index: int,
        tb_writer: SummaryWriter,
    ) -> float:
        running_loss = 0.0

        for i, data in enumerate(loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                logger.info(f"  batch {i + 1} loss: {running_loss / 1000:.4f}")
                tb_x = epoch_index * len(loader) + i + 1
                tb_writer.add_scalar("Loss/train", running_loss / 1000, tb_x)
                running_loss = 0.0

        return running_loss / len(loader) if len(loader) > 0 else 0.0

    def _validate_one(
        self,
        model: Module,
        loader: DataLoader,
        loss_fn: Module,
    ) -> float:
        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        model.train()
        return val_loss / len(loader) if len(loader) > 0 else 0.0

    def train(
        self,
        model: Module,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: Optimizer,
        loss_fn: Module,
        epochs: int,
        writer_log_dir: str | Path,
    ) -> Module:
        writer = SummaryWriter(log_dir=writer_log_dir)
        model.train()

        logger.info("Testing model...")

        best_val_loss = float("inf")
        best_model_state = None
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            logger.info(f"EPOCH {epoch + 1}/{epochs}")

            avg_loss = self._train_one(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epoch_index=epoch,
                tb_writer=writer,
            )

            val_loss = self._validate_one(
                model=model, loader=val_loader, loss_fn=loss_fn
            )
            logger.info(
                f"Epoch {epoch + 1} complete, avg loss: {avg_loss:.4f}, val loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                logger.info(f"New best validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            writer.add_scalar("Loss/epoch", avg_loss, epoch + 1)
            writer.add_scalar("Loss/val_epoch", val_loss, epoch + 1)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info("Loaded best model state from validation")

        writer.close()
        return model

    def test(
        self,
        model: Module,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ) -> Result:
        model.eval()

        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        metrics = Result(
            accuracy=accuracy_score(all_labels, all_preds),
            f1_macro=f1_score(all_labels, all_preds, average="macro", zero_division=0),
            precision_macro=precision_score(
                all_labels, all_preds, average="macro", zero_division=0
            ),
            recall_macro=recall_score(
                all_labels, all_preds, average="macro", zero_division=0
            ),
            classification_report=classification_report(
                all_labels, all_preds, digits=4, output_dict=True, zero_division=0
            ),
            confusion_matrix=confusion_matrix(all_labels, all_preds).tolist(),
        )

        return metrics
