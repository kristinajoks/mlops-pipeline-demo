FROM apache/airflow:3.2.0

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow