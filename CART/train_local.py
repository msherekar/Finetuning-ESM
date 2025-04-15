import time
import typer
import torch
import torch.nn as nn
import pytorch_lightning as pl
import json
import functools
import datetime
import numpy as np

from typing import Optional
from typing_extensions import Annotated

from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from .utils import collate_fn, get_run_id, get_loss_func, count_trainable_parameters
from .models import FinetuneESM, ESMLightningModule
from .lora import apply_lora, freeze_all_but_head
from .config import STORAGE_DIR, MLFLOW_TRACKING_URI, logger
from .data import load_data, CustomPreprocessor

app = typer.Typer()


@app.command()
def train_model(
    experiment_name: Annotated[str, typer.Option()],
    dataset_loc: Annotated[str, typer.Option()],
    targets_loc: Annotated[str, typer.Option()],
    task_type: Annotated[str, typer.Option()],
    num_samples: Annotated[Optional[int], typer.Option()] = None,
    val_size: Annotated[float, typer.Option()] = 0.2,
    esm_model: Annotated[str, typer.Option()] = "facebook/esm2_t6_8M_UR50D",
    dropout_p: Annotated[float, typer.Option()] = 0.05,
    num_classes: Annotated[int, typer.Option()] = 1,
    learning_rate: Annotated[float, typer.Option()] = 1e-3,
    training_mode: Annotated[str, typer.Option()] = "all_layers",
    num_epochs: Annotated[int, typer.Option()] = 20,
    batch_size: Annotated[int, typer.Option()] = 8,
    splits_json: Annotated[Optional[str], typer.Option()] = None,
    fold: Annotated[int, typer.Option()] = 0,
    verbose: Annotated[bool, typer.Option()] = True,
):
    # Seed everything
    ds = load_data(dataset_loc, num_samples)
    targets = np.load(targets_loc)

    if splits_json:
        with open(splits_json) as f:
            split = json.load(f)[fold]
        train_ds = ds.take(split["train_idx"])
        val_ds = ds.take(split["val_idx"])
    else:
        train_ds, val_ds = ds.train_test_split(val_size)

    
    preprocessor = CustomPreprocessor(esm_model, targets, task_type)
    train_ds = preprocessor.transform(train_ds).materialize()
    val_ds = preprocessor.transform(val_ds).materialize()

    num_train_samples = train_ds.count()
    tokenizer = AutoTokenizer.from_pretrained(esm_model)
    
    collate_fn_partial = functools.partial(collate_fn, tokenizer=tokenizer, task_type=task_type)
    train_loader = train_ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn_partial)
    val_loader = val_ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn_partial)

    # Model setup
    model = FinetuneESM(esm_model, dropout_p=dropout_p, num_classes=num_classes)

    if training_mode == "lora":
        apply_lora(model)
    elif training_mode == "head_only":
        freeze_all_but_head(model)

    if verbose:
        print(f"# Trainable Parameters: {count_trainable_parameters(model)}")

    loss_fn = get_loss_func(task_type)
    train_count = train_ds.count()
    total_steps = int(num_epochs * np.ceil(train_count / batch_size))

    # Score name logic
    if task_type == "regression":
        score_name = None
    elif task_type == "classification_binary":
        score_name = "binary_f1_score"
    elif task_type == "classification_multiclass":
        score_name = "multiclass_f1_score"
    elif task_type == "classification_multilabel":
        score_name = "multilabel_f1_score"
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    lightning_model = ESMLightningModule(
        model,
        total_steps,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        task_type=task_type
    )

    callbacks = []
    if score_name:
        callbacks.append(ModelCheckpoint(
            save_top_k=1,
            monitor="val_f1_score",
            mode="max",
            filename=f"{score_name}" + "-{epoch:02d}-{val_loss:.2f}"
        ))

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=MLFLOW_TRACKING_URI
    )
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",  # uses MPS on M1 or CPU fallback  
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=mlflow_logger
    )

    start = time.time()
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    end = time.time()

    print(f"✅ Training complete in {(end - start)/60:.2f} minutes")

    logger.info(json.dumps({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": get_run_id(experiment_name, "local"),
        "params": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        },
        "duration_min": (end - start) / 60
    }, indent=2))


if __name__ == "__main__":
    app()
