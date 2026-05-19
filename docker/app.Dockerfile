FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY streamlit_app.py ./
COPY data ./data
COPY reports/figures ./reports/figures

RUN uv sync --locked --no-dev

EXPOSE 8501

CMD ["uv", "run", "--no-sync", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
