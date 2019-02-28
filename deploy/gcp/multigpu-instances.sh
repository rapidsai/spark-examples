#!/usr/bin/env bash

# Override on the command line for different number of slaves.
: ${NUM_SLAVES:=4}

INSTANCES="mg-spark-master"
for ((i=0;i<NUM_SLAVES;i++)); do
  INSTANCES="${INSTANCES} mg-spark-slave-${i}"
done
