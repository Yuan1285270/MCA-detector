#!/bin/bash
cd /opt/arcticshift-q3
/usr/bin/python3 /opt/arcticshift-q3/arctic_q3_collector.py >> /opt/arcticshift-q3/logs/stdout.log 2>> /opt/arcticshift-q3/logs/stderr.log
