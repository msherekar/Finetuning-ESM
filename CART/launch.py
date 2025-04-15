# A Launcher script akin to launch.py at FDA
# A command line interface for training the model

import subprocess
import typer

app = typer.Typer()

@app.command()
def run(
    mode: str = typer.Option("debug", help="debug or distributed"),
    task_type: str = typer.Option("classification_binary"),
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
        cmd = ["python", "-m", "CART.train_debug"] + base_args
    elif mode == "distributed":
        cmd = ["python", "-m", "CART.train"] + base_args + [
            "--experiment-name", f"{task_type}-experiment",
            "--training-mode", "all_layers",
            "--num-workers", "1"
        ]
    else:
        typer.echo(f"❌ Unknown mode: {mode}")
        raise typer.Exit(code=1)

    typer.echo(f"🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    app()
