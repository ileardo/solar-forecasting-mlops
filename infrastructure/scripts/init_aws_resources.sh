#!/bin/bash
set -e

echo "🟡 Initializing LocalStack S3 resources..."

# Wait for LocalStack to be ready
echo "Waiting for LocalStack to start..."
for i in {1..30}; do
    if curl -s http://localhost:4566/_localstack/health >/dev/null 2>&1; then
        echo "✅ LocalStack is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ LocalStack failed to start within 30 seconds"
        exit 1
    fi
    sleep 1
done

# Create S3 bucket for MLflow artifacts
BUCKET_NAME=${S3_BUCKET_ARTIFACTS:-mlflow-artifacts}
echo "Creating S3 bucket: $BUCKET_NAME"

if awslocal s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "✅ Bucket '$BUCKET_NAME' already exists"
else
    awslocal s3 mb "s3://$BUCKET_NAME"
    echo "✅ Bucket '$BUCKET_NAME' created successfully"
fi

# List all buckets for verification
echo "📋 Current S3 buckets:"
awslocal s3 ls

echo "🎉 LocalStack initialization complete!"
