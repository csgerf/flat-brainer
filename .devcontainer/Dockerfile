# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim-buster

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Installing system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils gcc libpq-dev curl

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
