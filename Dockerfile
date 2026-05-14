FROM python:3.12.7-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.2 \
    fastapi==0.115.0 \
    uvicorn==0.30.6 \
    joblib==1.4.2

COPY MLProject/saved_model/ ./saved_model/
COPY MLProject/serve.py .

EXPOSE 8080

CMD ["python", "serve.py"]
