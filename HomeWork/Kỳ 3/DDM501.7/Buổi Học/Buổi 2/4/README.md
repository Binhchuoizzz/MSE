# TUTORIAL 04

Tutorial stack:

                        +------------------+
                        |   Client Apps    |
                        | (Python scripts) |
                        +------------------+
                                |
                                | HTTP/REST
                                v
                        +------------------+
                        | MLflow Tracking  |
                        |     Server       |
                        | (Port 5001)      |
                        +------------------+
                                |
                                +---------------+
                                |               |
                                v               v
                        +-------------+   +-------------+
                        | PostgreSQL  |   |   MinIO     |
                        | (Metadata)  |   | (Artifacts) |
                        | Port 5432   |   | Port 9000   |
                        +-------------+   +-------------+


## File .env

In order to run notebook in ./notebooks/

If run on docker change localhost to correct containers

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5001

# Model Configuration
MODEL_NAME=iris_classifier

# MinIO Configuration (for local notebook to upload artifacts)
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```

## Running Services

```bash
# Build và start tất cả services
docker compose up --build -d
```
To check the services:

- **MLflow UI**: http://localhost:5001
- **MinIO Console**: http://localhost:9001 (user: minioadmin / pass: minioadmin)