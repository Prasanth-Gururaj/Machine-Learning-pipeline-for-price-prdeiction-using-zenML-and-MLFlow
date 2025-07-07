import json
import numpy as np
import pandas as pd
import platform
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (str): The input data as a JSON string.

    Returns:
        np.ndarray: The model's prediction.
    """

    # ❌ Avoid calling service.start() on Windows to prevent daemon errors
    if platform.system() != "Windows":
        service.start(timeout=10)

    # Load the input data
    data = json.loads(input_data)

    # Clean up unwanted keys
    data.pop("columns", None)
    data.pop("index", None)

    expected_columns = [
        "Order", "PID", "MS SubClass", "Lot Frontage", "Lot Area", "Overall Qual",
        "Overall Cond", "Year Built", "Year Remod/Add", "Mas Vnr Area", "BsmtFin SF 1",
        "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF",
        "Low Qual Fin SF", "Gr Liv Area", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath",
        "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", "Fireplaces",
        "Garage Yr Blt", "Garage Cars", "Garage Area", "Wood Deck SF", "Open Porch SF",
        "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Pool Area", "Misc Val",
        "Mo Sold", "Yr Sold"
    ]

    df = pd.DataFrame(data["data"], columns=expected_columns)

    # Convert to JSON list
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # Run the prediction (calls the running MLflow server)
    prediction = service.predict(data_array)

    return prediction
