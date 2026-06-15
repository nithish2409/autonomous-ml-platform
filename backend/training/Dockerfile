FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir \
    minio \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    lightgbm \
    joblib

COPY train.py .

ENTRYPOINT ["python", "train.py"]
