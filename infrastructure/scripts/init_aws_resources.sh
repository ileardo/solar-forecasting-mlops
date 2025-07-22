#!/bin/bash

set -e

echo "🟡 Waiting for LocalStack to be ready..."
awslocal wait cloudformation-stack-create --stack-name "localstack-init" > /dev/null 2>&1 || true

echo "🟢 LocalStack is ready. Initializing resources..."

BUCKET_NAME=$(grep S3_BUCKET_ARTIFACTS .env | cut -d '=' -f2)

if awslocal s3 ls | grep -q "${BUCKET_NAME}"; then
  echo "Bucket '${BUCKET_NAME}' already exists."
else
  echo "Creating bucket '${BUCKET_NAME}'..."
  awslocal s3 mb "s3://${BUCKET_NAME}"
  echo "Bucket created."
fi

echo "✅ AWS resources initialized."
