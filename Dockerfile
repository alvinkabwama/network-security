FROM python:3.12-slim-bookworm
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN apt-get update && install -r requirements.txt
CMD ["python3", "app.py"]