Set-Location "C:\Users\krist\Downloads\Master\Master rad\mlops-pipeline-demo"
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000