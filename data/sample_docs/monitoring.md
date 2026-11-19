# Model Monitoring

Monitoring watches three things: latency, prediction distribution and
data drift. Latency is summarised with p95 over a rolling window.
Prediction distribution compares the live positive rate with the rate
seen in training data. Drift compares the mean of each live input
feature with its training mean; a feature is flagged when its
standardised shift exceeds 0.5. Drift usually means the upstream data
changed or user behaviour shifted, and the model may need retraining.
