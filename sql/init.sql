-- Solar Forecasting MLOps - Minimal Database Initialization
-- This script creates ONLY monitoring tables - MLflow manages its own schema

-- Create monitoring schema (MLflow creates its own)
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Simple monitoring table for model performance metrics
CREATE TABLE IF NOT EXISTS monitoring.model_performance (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metadata JSONB DEFAULT '{}'
);

-- Simple table for drift detection results
CREATE TABLE IF NOT EXISTS monitoring.data_drift (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dataset_name VARCHAR(255) NOT NULL,
    feature_name VARCHAR(255) NOT NULL,
    drift_score FLOAT NOT NULL,
    drift_detected BOOLEAN DEFAULT FALSE
);

-- Basic indexes for performance
CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp
ON monitoring.model_performance(timestamp);

CREATE INDEX IF NOT EXISTS idx_data_drift_timestamp
ON monitoring.data_drift(timestamp);

-- Grant permissions
GRANT USAGE ON SCHEMA monitoring TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO postgres;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Minimal monitoring schema initialized successfully!';
    RAISE NOTICE 'MLflow will create and manage its own tables automatically.';
END $$;
