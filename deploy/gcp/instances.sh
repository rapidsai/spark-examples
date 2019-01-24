#!/usr/bin/env bash

# Override on the command line for different number of slaves.
: ${NUM_SLAVES:=19}

INSTANCES="spark-master"
for ((i=0;i<NUM_SLAVES;i++)); do
  INSTANCES="${INSTANCES} spark-slave-${i}"
done
