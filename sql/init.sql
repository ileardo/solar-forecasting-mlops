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

-- Reference data table for storing training dataset statistics
CREATE TABLE IF NOT EXISTS monitoring.reference_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50),

    -- Training data metadata
    training_samples INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,
    target_horizons INTEGER NOT NULL,
    collection_timestamp TIMESTAMP NOT NULL,

    -- Complete reference statistics (stored as JSONB for flexibility)
    feature_statistics JSONB NOT NULL,
    target_statistics JSONB NOT NULL,
    correlation_analysis JSONB DEFAULT '{}',
    data_quality JSONB DEFAULT '{}',
    model_metadata JSONB DEFAULT '{}',

    -- Simplified drift reference (for faster drift detection queries)
    drift_reference JSONB DEFAULT '{}',

    -- Ensure one reference per model version
    UNIQUE(model_name, model_version)
);

-- Basic indexes for performance
CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp
ON monitoring.model_performance(timestamp);

CREATE INDEX IF NOT EXISTS idx_data_drift_timestamp
ON monitoring.data_drift(timestamp);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_reference_data_model
ON monitoring.reference_data(model_name);

CREATE INDEX IF NOT EXISTS idx_reference_data_timestamp
ON monitoring.reference_data(timestamp);

CREATE INDEX IF NOT EXISTS idx_reference_data_model_version
ON monitoring.reference_data(model_name, model_version);

-- Grant permissions
GRANT USAGE ON SCHEMA monitoring TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO postgres;
GRANT ALL PRIVILEGES ON monitoring.reference_data TO postgres;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Minimal monitoring schema initialized successfully!';
    RAISE NOTICE 'MLflow will create and manage its own tables automatically.';
END $$;
