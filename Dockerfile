# Client Inquiry Agent web demo (Render free web service, Docker).
FROM python:3.12-slim

# uv, pinned. Keep this in step with the local version (`uv --version`) so it reads uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Dependencies first (cached layer), then the code.
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra web --no-install-project
COPY --chown=user src ./src
# The brand files the agent answers from. The project is installed in editable mode, so
# retrieval finds them at /home/user/app/public_brand (next to src/).
COPY --chown=user public_brand ./public_brand
RUN uv sync --frozen --no-dev --extra web

# Documentation only: Render routes to the port it sets in PORT; locally the app falls back to 8000.
EXPOSE 8000

# GEMINI_API_KEY (this project's own key, from the techbygrace-inquiry-demo Google project) and,
# optionally, CLIENT_AGENT_MODEL are set as environment variables in the Render dashboard; they
# are never baked into the image and no .env file is read.
# client-inquiry-web listens on $PORT (else 8000) with the access log off. Messages are never
# stored or logged.
CMD ["client-inquiry-web"]
