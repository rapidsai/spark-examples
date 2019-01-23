#!/usr/bin/env bash

set -ex

# Override on the command line for different number of slaves.
: ${NUM_SLAVES:=19}

#gcloud compute ssh spark-master \
#  --command="cd /opt/spark && ./sbin/start-master.sh && ./sbin/start-slave.sh spark://spark-master:7077"

for ((i=0;i<NUM_SLAVES;i++)); do
  gcloud compute ssh spark-slave-${i} \
    --command="cd /opt/spark && ./sbin/start-slave.sh spark://spark-master:7077"
done
