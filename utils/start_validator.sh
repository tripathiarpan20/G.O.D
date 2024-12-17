#!/bin/bash

if [ -z "$GRAFANA_USERNAME" ]; then
  GRAFANA_USERNAME=admin
  sed -i '/GRAFANA_USERNAME/d' .vali.env
  echo GRAFANA_USERNAME=$GRAFANA_USERNAME >> .vali.env
fi

if [ -z "$GRAFANA_PASSWORD" ]; then
  GRAFANA_PASSWORD=$(openssl rand -hex 16)
  sed -i '/GRAFANA_PASSWORD/d' .vali.env
  echo GRAFANA_PASSWORD=$GRAFANA_PASSWORD >> .vali.env
fi

# Read port from .vali.env
VALIDATOR_PORT=$(grep VALIDATOR_PORT .vali.env | cut -d '=' -f2)

# Delete old validator services
pm2 delete validator || true
pm2 delete validator_cycle || true

# Load variables from .vali.env
set -a # Automatically export all variables
source .vali.env
set +a # Stop automatic export

# Define additional OTEL environment variables
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4317"
export OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true"
export OTEL_PYTHON_LOG_CORRELATION="true"

# Start the validator service using opentelemetry-instrument with combined env vars
pm2 start \
    "opentelemetry-instrument \
    --logs_exporter otlp \
    --traces_exporter none \
    --metrics_exporter otlp \
    --service_name query_node \
    uvicorn \
    --factory validator.asgi:factory \
    --host 0.0.0.0 \
    --port ${VALIDATOR_PORT} \
    --env-file .vali.env" \
    --name validator

# Start the validator_cycle service using opentelemetry-instrument
pm2 start \
    "opentelemetry-instrument \
    --logs_exporter otlp \
    --traces_exporter none \
    --metrics_exporter otlp \
    --service_name validator_cycle \
    python -u -m validator.cycle.main" \
    --name validator_cycle
