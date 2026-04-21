# Run Simple Test

## Prerequisite

- Python 3.12 or newer
- qoa4ml 0.3.x (install from source or `pip install qoa4ml`)

## Step run

1. Start Docker on the local machine.
2. Start RabbitMQ in Docker using `rabbitmq.sh`.
3. Start the report collector:

   ```bash
   python collector.py
   ```

4. Start a simple application instrumented with qoa4ml monitoring probes:

   ```bash
   python general_ml.py
   ```

## Simple configuration

All client configurations live in `./config/`. Each `clientN.yaml` exercises a stage of a five-client pipeline (gateway, data processing, two inference workers, and an aggregator).
