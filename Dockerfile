FROM python:3.12-slim

# Install Deno (required by dspy.RLM default Deno/Pyodide WASM sandbox)
ENV DENO_INSTALL="/usr/local"
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unzip ca-certificates \
    && deno_installer="$(mktemp)" \
    && trap 'rm -f "$deno_installer"' EXIT \
    && curl --fail --silent --show-error --location \
        --output "$deno_installer" https://deno.land/install.sh \
    && sh "$deno_installer" \
    && rm -f "$deno_installer" \
    && trap - EXIT \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .
USER appuser

EXPOSE 8080

CMD ["python", "app.py"]
