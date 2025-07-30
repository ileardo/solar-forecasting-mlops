"""
MLflow Model Registry manager for solar forecasting models.

This module provides comprehensive model registry management including
model registration, stage transitions, metadata management, and
model/preprocessor package handling for production deployment.
"""

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from mlflow.exceptions import MlflowException
from sklearn.multioutput import MultiOutputRegressor

from src.data.preprocessor import SolarForecastingPreprocessor
from src.utils.mlflow_utils import get_mlflow_client, setup_mlflow_tracking


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Comprehensive MLflow Model Registry manager for solar forecasting models.

    This class provides complete model registry functionality including:
    - Model + preprocessor package registration
    - Stage management (None -> Staging -> Production -> Archived)
    - Metadata and tag management
    - Model loading with preprocessor
    - Version comparison and analytics
    - Production deployment utilities

    The registry treats models and preprocessors as unified packages to ensure
    consistency between training and inference phases.

    Example:
        >>> registry = ModelRegistry()
        >>> version = registry.register_model_package(
        ...     model, preprocessor, "solar-forecasting-prod"
        ... )
        >>> registry.transition_to_production("solar-forecasting-prod", version)
        >>> loaded_model, loaded_preprocessor = registry.load_production_model(
        ...     "solar-forecasting-prod"
        ... )
    """

    def __init__(self) -> None:
        """Initialize the model registry manager."""
        # Setup MLflow tracking
        setup_mlflow_tracking()
        self.client = get_mlflow_client()

        logger.info("ModelRegistry initialized with MLflow backend")

    def register_model_package(
        self,
        model: MultiOutputRegressor,
        preprocessor: SolarForecastingPreprocessor,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """
        Register a complete model package (model + preprocessor) in MLflow registry.

        This version maintains separate storage: model in MLflow/S3, preprocessor in local artifacts directory.
        """
        logger.info(f"Registering model package: {model_name}")

        try:
            # Get current run ID if not provided
            if run_id is None:
                current_run = mlflow.active_run()
                if current_run is None:
                    raise RuntimeError("No active run found and no run_id provided")
                run_id = current_run.info.run_id

            # Create registered model if it doesn't exist
            try:
                self.client.create_registered_model(
                    name=model_name,
                    description=f"Solar forecasting model package created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                )
                logger.info(f"Created new registered model: {model_name}")
            except MlflowException as e:
                if "already exists" in str(e).lower():
                    logger.info(f"Using existing registered model: {model_name}")
                else:
                    raise

            # Save preprocessor to local artifacts directory
            project_root = Path(__file__).parent.parent.parent
            artifacts_dir = project_root / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            preprocessor_local_path = artifacts_dir / f"preprocessor_{run_id}.pkl"

            preprocessor.save_preprocessor(str(preprocessor_local_path))
            logger.info(f"Preprocessor saved locally at: {preprocessor_local_path}")

            # Register the EXISTING model from the run (don't re-log it)
            model_uri = f"runs:/{run_id}/model"

            # Create model version using existing model artifacts
            model_version_info = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                description=description
                or f"Solar forecasting model v{datetime.now().strftime('%Y%m%d_%H%M%S')} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )

            model_version = model_version_info.version
            logger.info(
                f"Registered existing model from {model_uri} as version {model_version}"
            )

            # Add tags with explicit run_id storage
            default_tags = {
                "model_type": "XGBoost MultiOutputRegressor",
                "forecast_horizon": str(preprocessor.forecast_horizon),
                "features_count": str(len(preprocessor.get_feature_names())),
                "created_date": datetime.now().strftime("%Y-%m-%d"),
                "mlops_stage": "development",
                "preprocessor_location": "local_artifacts",
                "source_run_id": run_id,
            }

            if tags:
                default_tags.update(tags)

            for key, value in default_tags.items():
                self.client.set_model_version_tag(
                    name=model_name, version=model_version, key=key, value=value
                )

            logger.info(
                f"Successfully registered model package {model_name} version {model_version}"
            )
            logger.info(f"Model URI: models:/{model_name}/{model_version}")
            logger.info(f"Preprocessor location: {preprocessor_local_path}")
            logger.info(f"Source run_id saved in tags: {run_id}")

            return model_version

        except Exception as e:
            logger.error(f"Failed to register model package: {str(e)}")
            raise RuntimeError(f"Model registration failed: {str(e)}") from e

    def load_model_package(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Tuple[MultiOutputRegressor, SolarForecastingPreprocessor]:
        """
        Load a complete model package (model + preprocessor) from the registry.
        """
        logger.info(f"Loading model package: {model_name}")

        try:
            # Determine model URI and get version info
            if version:
                model_uri = f"models:/{model_name}/{version}"
                logger.info(f"Loading specific version: {version}")
                model_version_info = self.client.get_model_version(model_name, version)
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
                logger.info(f"Loading from stage: {stage}")
                latest_versions = self.client.get_latest_versions(
                    model_name, stages=[stage]
                )
                if not latest_versions:
                    raise RuntimeError(f"No model found in stage {stage}")
                model_version_info = latest_versions[0]
            else:
                # Default to latest version
                latest_versions = self.client.get_latest_versions(
                    model_name, stages=["None"]
                )
                if not latest_versions:
                    raise RuntimeError(f"No versions found for model {model_name}")
                model_version_info = latest_versions[0]
                model_uri = f"models:/{model_name}/{model_version_info.version}"
                logger.info(f"Loading latest version: {model_version_info.version}")

            # Load the model
            logger.info(f"Loading model from URI: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)

            # Get run_id with fallback to tags
            run_id = model_version_info.run_id

            if not run_id or run_id.strip() == "":
                # Fallback: get run_id from tags
                logger.info("Attempting to retrieve run_id from tags...")

                # Handle different tag formats
                try:
                    if hasattr(model_version_info.tags, "__iter__") and hasattr(
                        list(model_version_info.tags)[0], "key"
                    ):
                        # Tags are objects with .key and .value attributes
                        tags = {tag.key: tag.value for tag in model_version_info.tags}
                    else:
                        # Tags are already a dictionary or similar structure
                        tags = dict(model_version_info.tags)

                    logger.info(f"Available tags: {list(tags.keys())}")
                    run_id = tags.get("source_run_id")

                except Exception as tag_error:
                    logger.error(f"Error processing tags: {tag_error}")
                    logger.info(f"Tags type: {type(model_version_info.tags)}")
                    logger.info(f"Tags content: {model_version_info.tags}")

                    # Last resort: try direct access if tags is a dict
                    try:
                        if isinstance(model_version_info.tags, dict):
                            run_id = model_version_info.tags.get("source_run_id")
                        else:
                            raise RuntimeError("Cannot parse tags structure")
                    except Exception:
                        raise RuntimeError(
                            f"Cannot determine run_id from tags. Tags structure: {type(model_version_info.tags)}"
                        )

                logger.info(f"Retrieved run_id from tags: {run_id}")

                if not run_id:
                    raise RuntimeError(
                        f"Cannot determine run_id for model version {model_version_info.version}"
                    )
            else:
                logger.info(f"Using run_id from model version info: {run_id}")

            # Load preprocessor from local artifacts directory
            project_root = Path(__file__).parent.parent.parent
            artifacts_dir = project_root / "artifacts"
            preprocessor_file = artifacts_dir / f"preprocessor_{run_id}.pkl"

            logger.info(f"Looking for preprocessor at: {preprocessor_file}")

            if not preprocessor_file.exists():
                raise RuntimeError(f"Preprocessor file not found: {preprocessor_file}")

            logger.info(f"Found preprocessor file: {preprocessor_file.name}")

            # Load the preprocessor
            logger.info(f"Loading preprocessor from: {preprocessor_file}")
            preprocessor = SolarForecastingPreprocessor.load_preprocessor(
                str(preprocessor_file)
            )

            logger.info(f"Successfully loaded model package {model_name}")
            logger.info(f"Model type: {type(model).__name__}")
            logger.info(
                f"Preprocessor features: {len(preprocessor.get_feature_names())}"
            )

            return model, preprocessor

        except Exception as e:
            logger.error(f"Failed to load model package: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e

    def transition_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = True,
    ) -> None:
        """
        Transition a model version to a specific stage.

        Args:
            model_name: Name of the registered model.
            version: Version to transition.
            stage: Target stage ('Staging', 'Production', 'Archived').
            archive_existing_versions: Whether to archive existing versions in target stage.

        Raises:
            RuntimeError: When stage transition fails.

        Example:
            >>> registry.transition_stage(
            ...     "solar-forecasting-prod", "2", "Production"
            ... )
        """
        logger.info(f"Transitioning model {model_name} v{version} to {stage}")

        try:
            # Archive existing versions in target stage if requested
            if archive_existing_versions and stage in ["Staging", "Production"]:
                existing_versions = self.client.get_latest_versions(
                    model_name, stages=[stage]
                )
                for existing_version in existing_versions:
                    logger.info(
                        f"Archiving existing version {existing_version.version} from {stage}"
                    )
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=existing_version.version,
                        stage="Archived",
                    )

            # Transition the new version
            self.client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )

            # Update stage-related tags
            self.client.set_model_version_tag(
                name=model_name, version=version, key="mlops_stage", value=stage.lower()
            )

            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key="promoted_date",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            logger.info(f"Successfully transitioned model to {stage}")

        except Exception as e:
            logger.error(f"Failed to transition model stage: {str(e)}")
            raise RuntimeError(f"Stage transition failed: {str(e)}") from e

    def transition_to_staging(self, model_name: str, version: str) -> None:
        """
        Convenience method to transition model to Staging.

        Args:
            model_name: Name of the registered model.
            version: Version to transition.
        """
        self.transition_stage(model_name, version, "Staging")

    def transition_to_production(self, model_name: str, version: str) -> None:
        """
        Convenience method to transition model to Production.

        Args:
            model_name: Name of the registered model.
            version: Version to transition.
        """
        self.transition_stage(model_name, version, "Production")

    def load_production_model(
        self, model_name: str
    ) -> Tuple[MultiOutputRegressor, SolarForecastingPreprocessor]:
        """
        Convenience method to load the current production model.

        Args:
            model_name: Name of the registered model.

        Returns:
            Tuple containing the production model and preprocessor.
        """
        return self.load_model_package(model_name, stage="Production")

    def load_staging_model(
        self, model_name: str
    ) -> Tuple[MultiOutputRegressor, SolarForecastingPreprocessor]:
        """
        Convenience method to load the current staging model.

        Args:
            model_name: Name of the registered model.

        Returns:
            Tuple containing the staging model and preprocessor.
        """
        return self.load_model_package(model_name, stage="Staging")

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a registered model.

        Args:
            model_name: Name of the registered model.

        Returns:
            Dict[str, Any]: Complete model information including all versions.
        """
        logger.info(f"Getting model info for: {model_name}")

        try:
            # Get registered model info
            registered_model = self.client.get_registered_model(model_name)

            # Get all versions
            versions = self.client.search_model_versions(f"name='{model_name}'")

            # Organize version information
            version_info = []
            for version in versions:
                tags = {tag.key: tag.value for tag in version.tags}

                version_info.append(
                    {
                        "version": version.version,
                        "stage": version.current_stage,
                        "creation_timestamp": version.creation_timestamp,
                        "last_updated_timestamp": version.last_updated_timestamp,
                        "run_id": version.run_id,
                        "description": version.description,
                        "tags": tags,
                        "status": version.status,
                    }
                )

            # Sort versions by version number
            version_info.sort(key=lambda x: int(x["version"]), reverse=True)

            # Get current stage versions
            current_stages = {}
            for stage in ["Production", "Staging", "None", "Archived"]:
                stage_versions = self.client.get_latest_versions(
                    model_name, stages=[stage]
                )
                if stage_versions:
                    current_stages[stage] = stage_versions[0].version

            model_info = {
                "name": registered_model.name,
                "description": registered_model.description,
                "creation_timestamp": registered_model.creation_timestamp,
                "last_updated_timestamp": registered_model.last_updated_timestamp,
                "total_versions": len(version_info),
                "current_stages": current_stages,
                "versions": version_info,
                "latest_version": version_info[0]["version"] if version_info else None,
            }

            logger.info(
                f"Retrieved info for model {model_name}: {len(version_info)} versions"
            )
            return model_info

        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            raise RuntimeError(f"Failed to get model info: {str(e)}") from e

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models with summary information.

        Returns:
            List[Dict[str, Any]]: List of model summaries.
        """
        logger.info("Listing all registered models...")

        try:
            registered_models = self.client.search_registered_models()

            model_list = []
            for model in registered_models:
                # Get latest versions for each stage
                current_stages = {}
                for stage in ["Production", "Staging", "None", "Archived"]:
                    stage_versions = self.client.get_latest_versions(
                        model.name, stages=[stage]
                    )
                    if stage_versions:
                        current_stages[stage] = stage_versions[0].version

                # Get total version count
                versions = self.client.search_model_versions(f"name='{model.name}'")

                model_summary = {
                    "name": model.name,
                    "description": model.description,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "total_versions": len(versions),
                    "current_stages": current_stages,
                }

                model_list.append(model_summary)

            logger.info(f"Found {len(model_list)} registered models")
            return model_list

        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            raise RuntimeError(f"Failed to list models: {str(e)}") from e

    def delete_model_version(self, model_name: str, version: str) -> None:
        """
        Delete a specific model version.

        Args:
            model_name: Name of the registered model.
            version: Version to delete.

        Raises:
            RuntimeError: When deletion fails.
        """
        logger.warning(f"Deleting model version {model_name} v{version}")

        try:
            self.client.delete_model_version(name=model_name, version=version)
            logger.info(f"Successfully deleted model version {model_name} v{version}")

        except Exception as e:
            logger.error(f"Failed to delete model version: {str(e)}")
            raise RuntimeError(f"Model version deletion failed: {str(e)}") from e

    def archive_old_versions(
        self, model_name: str, keep_latest_n: int = 5
    ) -> List[str]:
        """
        Archive old model versions, keeping only the latest N versions.

        Args:
            model_name: Name of the registered model.
            keep_latest_n: Number of latest versions to keep active.

        Returns:
            List[str]: List of archived version numbers.
        """
        logger.info(
            f"Archiving old versions for {model_name}, keeping latest {keep_latest_n}"
        )

        try:
            # Get all versions sorted by version number
            versions = self.client.search_model_versions(f"name='{model_name}'")
            versions.sort(key=lambda x: int(x.version), reverse=True)

            archived_versions = []

            # Archive versions beyond the keep limit
            for version in versions[keep_latest_n:]:
                if version.current_stage not in ["Production", "Staging", "Archived"]:
                    self.client.transition_model_version_stage(
                        name=model_name, version=version.version, stage="Archived"
                    )
                    archived_versions.append(version.version)
                    logger.info(f"Archived version {version.version}")

            logger.info(f"Archived {len(archived_versions)} old versions")
            return archived_versions

        except Exception as e:
            logger.error(f"Failed to archive old versions: {str(e)}")
            raise RuntimeError(f"Version archiving failed: {str(e)}") from e


# Export main class
__all__ = ["ModelRegistry"]
