#!/bin/bash

echo "Creating MLflow database..."

# Create MLflow database if it doesn't exist
psql -v ON_ERROR_STOP=0 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
    CREATE DATABASE mlflow;
EOSQL

echo "MLflow database setup completed"

