#!/bin/bash

start-master.sh
start-slave.sh spark://$HOSTNAME:7077
