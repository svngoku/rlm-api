FROM python:3.12-slim

# Install Deno (required by dspy.RLM default Deno/Pyodide WASM sandbox)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unzip ca-certificates \
    && curl -fsSL https://deno.land/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV DENO_INSTALL="/root/.deno"
ENV PATH="$DENO_INSTALL/bin:$PATH"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
