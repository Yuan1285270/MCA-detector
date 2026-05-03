#!/bin/bash
cd /opt/arcticshift-q1
/usr/bin/python3 /opt/arcticshift-q1/arctic_q1_collector.py >> /opt/arcticshift-q1/logs/stdout.log 2>> /opt/arcticshift-q1/logs/stderr.log
