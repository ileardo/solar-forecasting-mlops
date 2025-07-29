import os
import sys


project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.preprocessor import SolarForecastingPreprocessor
from src.model.evaluator import ModelEvaluator
from src.model.model_config import get_training_config
from src.model.registry import ModelRegistry
from src.model.trainer import ModelTrainer


# load preprocesso
temp_preprocessor = SolarForecastingPreprocessor(
    forecast_horizon=24,
    lag_days=[1, 2, 3, 7, 30],
    rolling_windows=[7, 30],
    scaling_method="standard",
)

# load data
generation_path = "./data/raw/Plant_1_Generation_Data.csv"
weather_path = "./data/raw/Plant_1_Weather_Sensor_Data.csv"
X_full, y_full, metadata = temp_preprocessor.fit_transform(
    generation_path, weather_path
)

# split data 80/20
split_point = int(0.8 * len(X_full))
X_test = X_full.iloc[split_point:].reset_index(drop=True)
y_test = y_full.iloc[split_point:].reset_index(drop=True)

# training config
config = get_training_config(generation_data_path=None, weather_data_path=None)

# train
trainer = ModelTrainer(config)
model, metrics, run_id = trainer.train()
preprocessor = trainer.preprocessor

# evaluate
evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model, X_test, y_test, preprocessor)
print("\n" + evaluator.get_evaluation_summary(results))

# registry
registry = ModelRegistry()
version = registry.register_model_package(
    model,
    preprocessor,
    "solar-forecasting-prod",
    description="XGBoost time series model with optimal parameters",
    tags={
        "algorithm": "xgboost",
        "horizon": "24h",
        "features": str(len(preprocessor.get_feature_names())),
        "test_rmse": str(round(results["overall"]["rmse"], 2)),
    },
    run_id=run_id,
)

# move to production
registry.transition_to_production("solar-forecasting-prod", version)
