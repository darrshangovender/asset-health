# --- build stage: install the package into an isolated prefix -----------------
FROM python:3.12-slim AS build

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY asset_health ./asset_health
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install .

# --- runtime stage: slim image, non-root, datasets live in /data -------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system app && useradd --system --gid app --create-home app \
    && mkdir -p /data && chown app:app /data

COPY --from=build /install /usr/local
WORKDIR /data
USER app
VOLUME ["/data"]

ENTRYPOINT ["asset-health"]
CMD ["--help"]
