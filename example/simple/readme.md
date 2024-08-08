# Run Simple Test

## Prerequisite
- Python > 3.10
- qoa4ml > 0.2.9

## Step run
1. Start docker service on local computer
2. Start RabbitMQ on local docker using `rabbitmq.sh`
3. Start report collector:
```bash
$ python collector.py
```
3. Start a simple application which is integrated qoa4ml monitoring probes:
```bash
$ python general_ml.py
```

## Simple configuration
All configurations are in `./config/` folder