#!/bin/bash
cd /opt/arcticshift-q4
/usr/bin/python3 /opt/arcticshift-q4/arctic_q4_collector.py >> /opt/arcticshift-q4/logs/stdout.log 2>> /opt/arcticshift-q4/logs/stderr.log
