import mlflow
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional
from transformers import EsmModel
from torchmetrics.functional.classification import (
    multilabel_f1_score,
    binary_f1_score,
    multiclass_f1_score,
    binary_auroc,
    binary_precision,
    binary_recall,
    binary_accuracy,
    multiclass_accuracy,
    binary_average_precision,
    multiclass_cohen_kappa,
    multilabel_auroc,
    multilabel_accuracy,
    multilabel_average_precision,
    multiclass_precision,
    multiclass_recall,
    multilabel_hamming_distance,
)
from torchmetrics.functional.regression import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


class FinetuneESM(nn.Module):
    def __init__(self, esm_model: str, dropout_p: float, num_classes: int, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.llm = EsmModel.from_pretrained(esm_model)
        if use_gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
        hidden = self.llm.config.hidden_size

        self.dropout = nn.Dropout(dropout_p)
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden, num_classes)
        self.num_classes = num_classes

    def mean_pooling(self, token_embeddings, attention_mask):
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
        sum_embeddings = torch.sum(token_embeddings * expanded_mask, dim=1)
        token_counts = torch.clamp(expanded_mask.sum(1), min=1e-9)
        return sum_embeddings / token_counts

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.llm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        pooled = self.mean_pooling(output.last_hidden_state, batch["attention_mask"])
        x = self.activation(self.pre_classifier(pooled))
        x = self.dropout(x)
        return self.classifier(x)


class ESMLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: FinetuneESM,
        total_steps: int,
        task_type: str,
        learning_rate: float = 1e-3,
        loss_fn: Optional[nn.Module] = None,
        class_labels: Optional[list] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.task_type = task_type
        
        # Store custom class labels if provided, otherwise use defaults
        self.class_labels = class_labels
        
        # Optional: Save hyperparameters to checkpoint
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch["targets"])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch["targets"])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        preds = logits
        targets = batch["targets"]

        # Track metrics based on task type
        if self.task_type == "regression":
            # Regression metrics
            mse = mean_squared_error(preds, targets)
            mae = mean_absolute_error(preds, targets)
            r2 = r2_score(preds, targets)
            
            self.log("val_mse", mse, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_r2", r2, prog_bar=True, on_step=False, on_epoch=True)
            
        elif self.task_type == "classification_binary":
            # Binary classification metrics
            probs = preds.sigmoid()
            score = binary_f1_score(probs, targets.int())
            accuracy = binary_accuracy(probs, targets.int())
            auc = binary_auroc(probs, targets.int())
            precision = binary_precision(probs, targets.int())
            recall = binary_recall(probs, targets.int())
            pr_auc = binary_average_precision(probs, targets.int())
            
            # Store predictions for confusion matrix
            if batch_idx == 0:  # Only on first batch to avoid overhead
                self.predictions = (probs > 0.5).float()
                self.true_labels = targets.int()
            else:
                self.predictions = torch.cat([self.predictions, (probs > 0.5).float()])
                self.true_labels = torch.cat([self.true_labels, targets.int()])
            
            self.log("val_f1_score", score, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_auc", auc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_precision", precision, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_recall", recall, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_pr_auc", pr_auc, prog_bar=True, on_step=False, on_epoch=True)
            
        elif self.task_type == "classification_multiclass":
            # Multiclass classification metrics
            probs = preds.softmax(dim=-1)
            score = multiclass_f1_score(
                probs, targets.long(),
                num_classes=self.model.num_classes,
                average="macro"
            )
            f1_micro = multiclass_f1_score(
                probs, targets.long(),
                num_classes=self.model.num_classes,
                average="micro"
            )
            f1_weighted = multiclass_f1_score(
                probs, targets.long(),
                num_classes=self.model.num_classes,
                average="weighted"
            )
            accuracy = multiclass_accuracy(
                probs, targets.long(),
                num_classes=self.model.num_classes,
                average="micro"
            )
            # Cohen's Kappa - agreement between predictions and ground truth
            kappa = multiclass_cohen_kappa(
                probs, targets.long(),
                num_classes=self.model.num_classes
            )
            
            # Per-class precision and recall
            precision_per_class = multiclass_precision(
                probs, targets.long(),
                num_classes=self.model.num_classes,
                average=None
            )
            recall_per_class = multiclass_recall(
                probs, targets.long(),
                num_classes=self.model.num_classes,
                average=None
            )
            
            # Log the metrics
            self.log("val_f1_score", score, prog_bar=True, on_step=False, on_epoch=True)  # Macro F1
            self.log("val_f1_micro", f1_micro, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_f1_weighted", f1_weighted, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_cohen_kappa", kappa, prog_bar=True, on_step=False, on_epoch=True)
            
            # Log per-class metrics
            for i in range(self.model.num_classes):
                self.log(f"val_precision_class_{i}", precision_per_class[i], prog_bar=True, on_step=False, on_epoch=True)
                self.log(f"val_recall_class_{i}", recall_per_class[i], prog_bar=True, on_step=False, on_epoch=True  )
            
            # Confusion matrix is typically logged as an artifact rather than a metric
            # We'll compute and log predictions for later confusion matrix visualization
            if batch_idx == 0:  # Only on first batch to avoid overhead
                self.predictions = probs.argmax(dim=1)
                self.true_labels = targets.long()
            else:
                self.predictions = torch.cat([self.predictions, probs.argmax(dim=1)])
                self.true_labels = torch.cat([self.true_labels, targets.long()])
            
        elif self.task_type == "classification_multilabel":
            # Multilabel classification metrics
            probs = preds.sigmoid()
            
            # F1 scores (macro, micro, weighted)
            f1_macro = multilabel_f1_score(
                probs, targets.int(), 
                num_labels=self.model.num_classes,
                average="macro"
            )
            f1_micro = multilabel_f1_score(
                probs, targets.int(), 
                num_labels=self.model.num_classes,
                average="micro"
            )
            f1_weighted = multilabel_f1_score(
                probs, targets.int(), 
                num_labels=self.model.num_classes,
                average="weighted"
            )
            
            # Subset accuracy (exact match)
            subset_accuracy = multilabel_accuracy(
                probs, targets.int(),
                num_labels=self.model.num_classes,
                threshold=0.5
            )
            
            # Hamming distance (fraction of wrong labels - lower is better)
            hamming_loss = multilabel_hamming_distance(
                probs, targets.int(),
                num_labels=self.model.num_classes,
                threshold=0.5
            )
            
            # ROC AUC scores (macro, micro)
            try:
                roc_auc_macro = multilabel_auroc(
                    probs, targets.int(),
                    num_labels=self.model.num_classes,
                    average="macro"
                )
                roc_auc_micro = multilabel_auroc(
                    probs, targets.int(),
                    num_labels=self.model.num_classes,
                    average="micro"
                )
                self.log("val_roc_auc_macro", roc_auc_macro, prog_bar=True)
                self.log("val_roc_auc_micro", roc_auc_micro, prog_bar=True)
            except Exception as e:
                # Sometimes AUROC can fail if a class isn't present in the batch
                self.log("val_roc_auc_error", 1.0)
            
            # PR-AUC score
            try:
                pr_auc = multilabel_average_precision(
                    probs, targets.int(),
                    num_labels=self.model.num_classes,
                    average="macro"
                )
                self.log("val_pr_auc", pr_auc, prog_bar=True)
            except Exception as e:
                # Sometimes PR-AUC can fail if a class isn't present in the batch
                self.log("val_pr_auc_error", 1.0)
            
            # Log the main metrics
            self.log("val_f1_score", f1_macro, prog_bar=True, on_step=False, on_epoch=True)  # Main benchmark metric
            self.log("val_f1_micro", f1_micro, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_f1_weighted", f1_weighted, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_subset_accuracy", subset_accuracy, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_hamming_loss", hamming_loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        # Create and log confusion matrix at the end of validation
        if hasattr(self, 'predictions') and hasattr(self, 'true_labels'):
            try:
                
                
                # Convert tensors to numpy
                preds_np = self.predictions.cpu().numpy()
                targets_np = self.true_labels.cpu().numpy()
                
                # Handle different task types
                if self.task_type == "classification_binary":
                    # Binary confusion matrix
                    cm = confusion_matrix(targets_np, preds_np)
                    
                    # Use custom class labels if provided, otherwise use defaults
                    if self.class_labels and len(self.class_labels) == 2:
                        class_labels = self.class_labels
                    else:
                        # Default to "0" and "1" instead of "Negative" and "Positive"
                        class_labels = ["0", "1"]
                        
                    title = "Binary Confusion Matrix"
                elif self.task_type == "classification_multiclass":
                    # Multiclass confusion matrix
                    cm = confusion_matrix(targets_np, preds_np)
                    
                    # Use custom class labels if provided, otherwise use defaults
                    if self.class_labels and len(self.class_labels) == self.model.num_classes:
                        class_labels = self.class_labels
                    else:
                        class_labels = [str(i) for i in range(self.model.num_classes)]
                        
                    title = "Multiclass Confusion Matrix"
                else:
                    # Skip for other task types
                    return
                
                # Create a figure with the confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=class_labels,
                    yticklabels=class_labels
                )
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(title)
                
                # Save the figure and log as artifact
                confusion_matrix_path = f"confusion_matrix_{self.task_type}_epoch_{self.current_epoch}.png"
                plt.savefig(confusion_matrix_path)
                plt.close()
                
                # Log using MLflow if available
                try:
                    
                    mlflow.log_artifact(confusion_matrix_path)
                except (ImportError, Exception) as e:
                    self.logger.experiment.add_figure(
                        f"confusion_matrix_{self.task_type}", 
                        plt.figure(figsize=(8, 6)), 
                        self.current_epoch
                    )
            except Exception as e:
                print(f"Failed to create confusion matrix: {e}")
            
            # Clear for next epoch
            del self.predictions
            del self.true_labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, total_steps=self.total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
