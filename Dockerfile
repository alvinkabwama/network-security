FROM python:3.12-slim-bookworm

WORKDIR /app
COPY . /app

# Optional: only if you truly need awscli inside the container
# otherwise, delete this whole RUN
RUN apt-get update -y && \
    apt-get install -y awscli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Start your app (adjust if your entrypoint is different)
CMD ["python3", "app.py"]
