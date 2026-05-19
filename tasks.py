import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "wahlumfragen"
PYTHON_VERSION = "3.12"
DOCKER_IMAGE = "wahlumfragen-app"
DOCKER_PORT = 8501


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
    """Build the Streamlit app Docker image."""
    ctx.run(
        f"docker build -f docker/app.Dockerfile -t {DOCKER_IMAGE}:latest . --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_run(ctx: Context, port: int = DOCKER_PORT) -> None:
    """Run the Streamlit app Docker container."""
    ctx.run(f"docker run --rm -p {port}:{DOCKER_PORT} {DOCKER_IMAGE}:latest", echo=True, pty=not WINDOWS)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
