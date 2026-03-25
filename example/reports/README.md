# QoA4ML Report Examples

These examples demonstrate how to use the QoA4ML reporting and contract APIs. All examples use the **debug connector**, so no external message broker (RabbitMQ, Kafka, etc.) is required.

## Examples

### `basic_report.py`

End-to-end example that:

1. Creates a `QoaClient` with a debug connector
2. Observes service metrics (response time, reliability, availability)
3. Observes data quality metrics (accuracy, completeness)
4. Observes inference metrics (prediction value and confidence)
5. Generates and prints the full quality report as JSON

Run:

```bash
cd example/reports
python basic_report.py
```

### `contract_example.py`

Shows how to define quality contracts using the constraint model:

1. Creates `Metric`, `Condition`, and `MetricConstraint` objects
2. Combines them into a `BaseConstraint` (a quality contract)
3. Serializes the contract to JSON and demonstrates round-trip parsing

Run:

```bash
cd example/reports
python contract_example.py
```

## Configuration

The `config/client.yaml` file configures the client with a debug connector. Edit this file to switch to AMQP or other connectors for production use.
