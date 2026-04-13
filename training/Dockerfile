FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    lightgbm \
    scikit-learn \
    mlflow \
    pandas \
    numpy \
    dvc \
    great-expectations

COPY src/ ./src/

CMD ["python", "-c", "print('Training container ready')"]