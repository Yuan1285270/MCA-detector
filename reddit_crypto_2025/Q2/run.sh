#!/bin/bash
cd /opt/arcticshift-q2
/usr/bin/python3 /opt/arcticshift-q2/arctic_q2_collector.py >> /opt/arcticshift-q2/logs/stdout.log 2>> /opt/arcticshift-q2/logs/stderr.log
