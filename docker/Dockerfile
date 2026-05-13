FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --uid 1000 amms \
    && mkdir -p /data \
    && chown -R amms:amms /data /app

COPY --chown=amms:amms pyproject.toml README.md ./
COPY --chown=amms:amms src/ ./src/

RUN pip install --no-cache-dir .

COPY --chown=amms:amms docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER amms

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["run"]
