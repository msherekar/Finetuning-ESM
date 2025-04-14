import os
import subprocess
import typer

app = typer.Typer()

@app.command()
def run(
    mode: str = typer.Option("debug", help="debug or distributed"),
    task_type: str = typer.Option("classification_multilabel"),
    model: str = typer.Option("facebook/esm2_t6_8M_UR50D"),
    target_path: str = typer.Option("targets.npy"),
    data_path: str = typer.Option("data.parquet"),
    num_classes: int = typer.Option(100),
):
    base_args = [
        "--dataset-loc", data_path,
        "--targets-loc", target_path,
        "--task-type", task_type,
        "--num-classes", str(num_classes),
        "--esm-model", model
    ]

    if mode == "debug":
        subprocess.run(["python", "train_debug.py", "train-debug"] + base_args)
    elif mode == "distributed":
        subprocess.run(["python", "train.py", "train-model"] + base_args + [
            "--experiment-name", f"{task_type}-experiment",
            "--training-mode", "all_layers",
            "--num-workers", "1"
        ])
    else:
        print("Unknown mode:", mode)

if __name__ == "__main__":
    app()
