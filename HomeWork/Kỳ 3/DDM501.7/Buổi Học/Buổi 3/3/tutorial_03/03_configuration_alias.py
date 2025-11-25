"""
    This script is to demonstrate how to configure the alias for the registered model.
    We will use the mlflow client to configure the alias for the registered model 
            and promote the model from one alias to another.
"""
import os
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient


load_dotenv(dotenv_path=".env")
MODEL_NAME = "breast_cancer-predictor"


def configure_alias(model_name: str, alias: str, version: str):
    """
    This function is to configure the alias for the registered model.
    Args:
        model_name: str
        alias: str
        version: str
    Returns:
        None
    """
    client = MlflowClient(tracking_uri=os.environ['MLFLOW_TRACKING_URI'])
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version
    )
    print(f'Configured alias {alias} for model {model_name} with version {version} successfully')
    return


def promote_model(model_name: str, from_alias: str, to_alias: str):
    """
    This function is to promote the model from one alias to another.
    Args:
        model_name: str
        from_alias: str
        to_alias: str
    Returns:
        version: str
    """
    client = MlflowClient(tracking_uri=os.environ['MLFLOW_TRACKING_URI'])
    from_version = client.get_model_version_by_alias(
                        name=model_name,
                        alias=from_alias
    )
    version = from_version.version
    print(f'Getting version {version} from alias {from_alias} for model {model_name}')

    client.set_registered_model_alias(
        name=model_name,
        alias=to_alias,
        version=version
    )
    print(f'Promoted model {model_name} from {from_alias} to {to_alias} successfully')
    return version

if __name__ == "__main__":
    configure_alias(MODEL_NAME, "dev", "5")
    configure_alias(MODEL_NAME, "staging", "6")
    configure_alias(MODEL_NAME, "prod", "7")

    version = promote_model(MODEL_NAME, "dev", "staging")
    print(f'Promoted model {MODEL_NAME} from dev to staging successfully with version {version}')

    # version = promote_model(MODEL_NAME, "staging", "prod")
    # print(f'Promoted model {MODEL_NAME} from staging to prod successfully with version {version}')