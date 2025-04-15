import time
import typer
import torch
import torch.nn as nn
import pytorch_lightning as pl
import json
import functools
import datetime
import numpy as np
import ray

from typing import Optional
from typing_extensions import Annotated

from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from .utils import collate_fn, get_run_id, get_loss_func, count_trainable_parameters, find_optimal_batch_size, get_optimal_epochs
from .models import FinetuneESM, ESMLightningModule
from .lora import apply_lora, apply_lora_multicore, freeze_all_but_head
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
    training_mode: Annotated[str, typer.Option(help="Training mode: all_layers | lora | multicore_lora | head_only")] = "all_layers",
    num_epochs: Annotated[Optional[int], typer.Option(help="Number of epochs (None for auto-determination)")] = None,
    batch_size: Annotated[Optional[int], typer.Option()] = None,
    splits_json: Annotated[Optional[str], typer.Option()] = None,
    fold: Annotated[int, typer.Option()] = 0,
    use_cpu_cores: Annotated[bool, typer.Option()] = False,
    benchmark: Annotated[bool, typer.Option(help="Run a quick benchmark of different training modes")] = False,
    verbose: Annotated[bool, typer.Option()] = True,
):
    """
    Train an ESM model for protein sequence tasks.
    
    Examples:
    
    Train with LoRA (fastest on MPS):
    python -m CART.train_local --experiment_name "esm_finetune" --dataset_loc "data/sequences.parquet" --targets_loc "data/targets.npy" --task_type "classification_binary" --num_classes 2 --training_mode "lora"
    
    Train with multicore CPU and LoRA:
    python -m CART.train_local --experiment_name "esm_finetune" --dataset_loc "data/sequences.parquet" --targets_loc "data/targets.npy" --task_type "classification_binary" --num_classes 2 --training_mode "multicore_lora" --use_cpu_cores
    
    Train using all cores (automatically uses optimized LoRA):
    python -m CART.train_local --experiment_name "esm_finetune" --dataset_loc "data/sequences.parquet" --targets_loc "data/targets.npy" --task_type "classification_binary" --num_classes 2 --use_cpu_cores
    
    Run a quick benchmark to find the fastest training configuration:
    python -m CART.train_local --experiment_name "esm_finetune" --dataset_loc "data/sequences.parquet" --targets_loc "data/targets.npy" --task_type "classification_binary" --num_classes 2 --benchmark
    """
    # Seed everything
    ds = load_data(dataset_loc, num_samples)
    targets = np.load(targets_loc)

    if splits_json:
        with open(splits_json) as f:
            split = json.load(f)[fold]
        # take() returns a list, so we need to convert back to Ray dataset
        train_ds = ray.data.from_items(ds.take(split["train_idx"]))
        val_ds = ray.data.from_items(ds.take(split["val_idx"]))
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

    # Get a sample batch to estimate optimal batch size
    if batch_size is None:
        sample_batch = next(iter(train_ds.iter_torch_batches(batch_size=1, collate_fn=collate_fn_partial)))
    
    # Model setup
    model = FinetuneESM(
        esm_model, 
        dropout_p=dropout_p, 
        num_classes=num_classes,
        use_gradient_checkpointing=True  # Enable gradient checkpointing
    )

    # Auto-determine batch size if not provided
    if batch_size is None and verbose:
        batch_size = find_optimal_batch_size(model, sample_batch)
        print(f"🔍 Auto-determined optimal batch size: {batch_size}")
        # Recreate dataloaders with optimal batch size
        train_loader = train_ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn_partial)
        val_loader = val_ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn_partial)

    # If benchmarking, test different configurations to find the fastest
    if benchmark:
        print("🧪 Running benchmark to determine fastest training configuration...")
        
        configs = [
            {"name": "MPS with LoRA", "mode": "lora", "cpu_cores": False, "epochs": 1},
            {"name": "CPU with multicore LoRA", "mode": "multicore_lora", "cpu_cores": True, "epochs": 1}
        ]
        
        results = []
        for config in configs:
            print(f"\n⏱️ Testing: {config['name']}...")
            
            # Apply configuration
            test_model = FinetuneESM(
                esm_model, 
                dropout_p=dropout_p,
                num_classes=num_classes,
                use_gradient_checkpointing=True
            )
            
            if config["mode"] == "lora":
                apply_lora(test_model)
            elif config["mode"] == "multicore_lora":
                apply_lora_multicore(test_model)
                
            # Setup benchmark trainer
            test_callbacks = []
            test_trainer = pl.Trainer(
                max_epochs=config["epochs"],
                accelerator="mps" if not config["cpu_cores"] else "cpu",
                devices=1 if not config["cpu_cores"] else "auto",
                strategy="auto",  # Use single-process strategy to avoid pickling issues
                callbacks=test_callbacks,
                logger=False,
                enable_progress_bar=True,
                enable_checkpointing=False,
                enable_model_summary=False,
            )
            
            # Measure time
            start_time = time.time()
            test_lightning_model = ESMLightningModule(
                test_model,
                total_steps=int(config["epochs"] * np.ceil(num_train_samples / batch_size)),
                learning_rate=learning_rate,
                loss_fn=get_loss_func(task_type),
                task_type=task_type
            )
            
            # Use a small subset of data for benchmarking
            # take() returns a list, so we need to convert back to a dataset
            benchmark_size = min(5000, num_train_samples)
            val_size = min(1000, val_ds.count())
            
            # Instead of using take(), use a different approach to create a subset
            # Create benchmark datasets from the sample rows
            benchmark_train = ray.data.from_items(train_ds.take(benchmark_size))
            benchmark_val = ray.data.from_items(val_ds.take(val_size))
            
            benchmark_train_loader = benchmark_train.iter_torch_batches(
                batch_size=batch_size, 
                collate_fn=collate_fn_partial
            )
            benchmark_val_loader = benchmark_val.iter_torch_batches(
                batch_size=batch_size, 
                collate_fn=collate_fn_partial
            )
            
            test_trainer.fit(
                test_lightning_model, 
                train_dataloaders=benchmark_train_loader,
                val_dataloaders=benchmark_val_loader
            )
            duration = time.time() - start_time
            
            results.append({
                "name": config["name"],
                "duration": duration,
                "mode": config["mode"],
                "cpu_cores": config["cpu_cores"]
            })
            print(f"⏱️ {config['name']} took {duration:.2f} seconds")
            
        # Find fastest configuration
        fastest = min(results, key=lambda x: x["duration"])
        print(f"\n🏆 Fastest configuration: {fastest['name']} ({fastest['duration']:.2f} seconds)")
        print(f"Recommended command: python -m CART.train_local --training_mode {fastest['mode']} {'--use_cpu_cores' if fastest['cpu_cores'] else ''}")
        
        # Apply the fastest configuration
        if fastest["mode"] != training_mode or fastest["cpu_cores"] != use_cpu_cores:
            print(f"\n🔄 Applying fastest configuration: {fastest['name']}")
            training_mode = fastest["mode"]
            use_cpu_cores = fastest["cpu_cores"]

    # Auto-switch to multicore_lora if using CPU cores with lora
    if use_cpu_cores and training_mode == "lora":
        print("📝 Auto-switching to multicore_lora for better performance with multiple cores")
        training_mode = "multicore_lora"
    
    if training_mode == "lora":
        apply_lora(model)
    elif training_mode == "head_only":
        freeze_all_but_head(model)
    elif training_mode == "multicore_lora":
        apply_lora_multicore(model)

    if verbose:
        print(f"# Trainable Parameters: {count_trainable_parameters(model)}")

    loss_fn = get_loss_func(task_type)

    # Get total dataset size for auto-configuration
    train_count = train_ds.count()
    
    # Auto-determine number of epochs if not specified
    if num_epochs is None:
        num_epochs = get_optimal_epochs(train_count, training_mode, task_type)
        if verbose:
            print(f"🔍 Auto-determined optimal number of epochs: {num_epochs}")
            
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
        
    # Add early stopping to automatically terminate training when metrics plateau
    callbacks.append(EarlyStopping(
        monitor="val_f1_score" if score_name else "val_loss",
        mode="max" if score_name else "min",
        patience=5,  # Stop after 5 epochs without improvement
        min_delta=0.001  # Minimum change to qualify as improvement
    ))

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=MLFLOW_TRACKING_URI
    )
    
    # Trainer setup
    if use_cpu_cores:
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="cpu",
            devices="auto",  # Use all available CPU cores
            strategy="auto",  # Use default strategy to avoid pickling issues with Ray datasets
            precision="16-mixed",  # Use mixed precision for faster training
            callbacks=callbacks,
            log_every_n_steps=1,
            logger=mlflow_logger
        )
    else:
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="mps",  # Specifically use MPS for Metal Performance Shaders
            devices=1,  # Use a single device (M1 GPU)
            strategy="auto",  # Let PyTorch Lightning choose the best strategy
            precision="16-mixed",  # Use mixed precision for faster training
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
