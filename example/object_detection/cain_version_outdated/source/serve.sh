#! /bin/bash

ray start --head --port 6379
sleep 2
python /home/source/deployment.py
tail -f /dev/null