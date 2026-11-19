# Deploying ML Models

Models in this system are deployed behind a FastAPI prediction API.
Every request is validated against a schema before inference, and every
prediction is logged with its input features, output and latency.
The API exposes a /health endpoint for load-balancer checks and a
/predict endpoint for scoring. For AWS deployments, the same code runs
as a Lambda container image or behind a SageMaker endpoint.
p95 latency is tracked continuously; the alert threshold is 250 ms.
