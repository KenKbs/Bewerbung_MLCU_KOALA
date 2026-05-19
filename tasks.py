import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "wahlumfragen"
PYTHON_VERSION = "3.12"


# Project commands
@task
def generate_data(ctx: Context) -> None:
    """Generate data and save the sample.csv do the data folder"""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/sample.csv", echo=True, pty=not WINDOWS)


@task
def simulate_polls(ctx: Context, draws: int = 50000) -> None:
    """Run MCMC simulation by first aggregating, then modeling uncertainty and drawing x samples"""
    ctx.run(f"uv run src/{PROJECT_NAME}/model.py --n-draws={draws}")


@task
def plot(ctx: Context, save_svg: bool = False) -> None:
    """Plot the results of the MCMC simulation."""
    save_svg_flag = " --save-svg" if save_svg else ""
    ctx.run(f"uv run src/{PROJECT_NAME}/visualize.py{save_svg_flag}", echo=True, pty=not WINDOWS)


@task
def app(ctx: Context) -> None:
    """Run the Streamlit dashboard."""
    ctx.run("uv run streamlit run streamlit_app.py", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
