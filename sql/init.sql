-- Solar Forecasting MLOps - PostgreSQL Initialization Script
-- This script sets up the database schema for MLflow and monitoring

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS mlflow;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS public;

-- Set default schema search path
ALTER DATABASE solar_forecasting SET search_path TO public, mlflow, monitoring;

-- Create monitoring tables for Evidently metrics
CREATE TABLE IF NOT EXISTS monitoring.model_performance (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50),
    dataset_name VARCHAR(255),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitoring.data_drift (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    dataset_name VARCHAR(255) NOT NULL,
    feature_name VARCHAR(255) NOT NULL,
    drift_score FLOAT NOT NULL,
    drift_detected BOOLEAN DEFAULT FALSE,
    p_value FLOAT,
    threshold_value FLOAT,
    drift_method VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitoring.prediction_monitoring (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(255) NOT NULL,
    prediction_id UUID DEFAULT uuid_generate_v4(),
    input_features JSONB NOT NULL,
    prediction FLOAT NOT NULL,
    actual_value FLOAT,
    error_value FLOAT,
    absolute_error FLOAT,
    squared_error FLOAT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitoring.batch_predictions (
    id SERIAL PRIMARY KEY,
    batch_id UUID DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50),
    input_data_path VARCHAR(500),
    output_data_path VARCHAR(500),
    num_predictions INTEGER,
    avg_prediction FLOAT,
    min_prediction FLOAT,
    max_prediction FLOAT,
    batch_metadata JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitoring.model_alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium',
    model_name VARCHAR(255),
    metric_name VARCHAR(100),
    current_value FLOAT,
    threshold_value FLOAT,
    message TEXT,
    status VARCHAR(20) DEFAULT 'active',
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON monitoring.model_performance(timestamp);
CREATE INDEX IF NOT EXISTS idx_model_performance_model ON monitoring.model_performance(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_data_drift_timestamp ON monitoring.data_drift(timestamp);
CREATE INDEX IF NOT EXISTS idx_data_drift_feature ON monitoring.data_drift(dataset_name, feature_name);
CREATE INDEX IF NOT EXISTS idx_prediction_monitoring_timestamp ON monitoring.prediction_monitoring(timestamp);
CREATE INDEX IF NOT EXISTS idx_prediction_monitoring_model ON monitoring.prediction_monitoring(model_name);
CREATE INDEX IF NOT EXISTS idx_batch_predictions_timestamp ON monitoring.batch_predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_model_alerts_timestamp ON monitoring.model_alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_model_alerts_status ON monitoring.model_alerts(status);

-- Create functions for data retention (optional cleanup)
CREATE OR REPLACE FUNCTION monitoring.cleanup_old_records(
    table_name TEXT,
    days_to_keep INTEGER DEFAULT 90
) RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    EXECUTE format(
        'DELETE FROM monitoring.%I WHERE created_at < NOW() - INTERVAL ''%s days''',
        table_name, days_to_keep
    );
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create views for common monitoring queries
CREATE OR REPLACE VIEW monitoring.latest_model_performance AS
SELECT DISTINCT ON (model_name, metric_name)
    model_name,
    model_version,
    metric_name,
    metric_value,
    timestamp,
    metadata
FROM monitoring.model_performance
ORDER BY model_name, metric_name, timestamp DESC;

CREATE OR REPLACE VIEW monitoring.active_drift_alerts AS
SELECT
    dataset_name,
    feature_name,
    drift_score,
    p_value,
    drift_method,
    timestamp
FROM monitoring.data_drift
WHERE drift_detected = TRUE
  AND timestamp > NOW() - INTERVAL '24 hours'
ORDER BY drift_score DESC;

CREATE OR REPLACE VIEW monitoring.recent_predictions AS
SELECT
    model_name,
    COUNT(*) as prediction_count,
    AVG(prediction) as avg_prediction,
    AVG(CASE WHEN actual_value IS NOT NULL THEN absolute_error END) as avg_absolute_error,
    MAX(timestamp) as last_prediction_time
FROM monitoring.prediction_monitoring
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY model_name
ORDER BY last_prediction_time DESC;

-- Grant permissions
GRANT USAGE ON SCHEMA monitoring TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO postgres;

-- Insert initial test data (optional)
INSERT INTO monitoring.model_performance (model_name, model_version, metric_name, metric_value) VALUES
    ('solar-baseline', '1.0', 'rmse', 45.67),
    ('solar-baseline', '1.0', 'mae', 32.45),
    ('solar-baseline', '1.0', 'r2_score', 0.85);

-- Create notification function for alerts (PostgreSQL NOTIFY)
CREATE OR REPLACE FUNCTION monitoring.notify_alert() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.severity IN ('high', 'critical') THEN
        PERFORM pg_notify(
            'model_alert',
            json_build_object(
                'id', NEW.id,
                'type', NEW.alert_type,
                'severity', NEW.severity,
                'model_name', NEW.model_name,
                'message', NEW.message
            )::text
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for alert notifications
DROP TRIGGER IF EXISTS alert_notification ON monitoring.model_alerts;
CREATE TRIGGER alert_notification
    AFTER INSERT ON monitoring.model_alerts
    FOR EACH ROW
    EXECUTE FUNCTION monitoring.notify_alert();

-- Final success message
DO $$
BEGIN
    RAISE NOTICE 'Solar Forecasting MLOps database initialized successfully!';
    RAISE NOTICE 'Schemas created: public, mlflow, monitoring';
    RAISE NOTICE 'Tables created: 5 monitoring tables with indexes and views';
END $$;
