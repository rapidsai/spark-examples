#!/usr/bin/env bash

set -ex

# Override on the command line for different number of slaves.
: ${NUM_SLAVES:=4}

gcloud compute scp deploy/gcp/start-slave.sh mg-spark-master:/opt/spark/sbin/
gcloud compute ssh mg-spark-master \
  --command="cd /opt/spark && ./sbin/start-master.sh && SPARK_WORKER_INSTANCES=4 ./sbin/start-slave.sh -c 16 -m 103G spark://mg-spark-master:7077"

for ((i=0;i<NUM_SLAVES;i++)); do
  gcloud compute scp deploy/gcp/start-slave.sh mg-spark-slave-${i}:/opt/spark/sbin/
  gcloud compute ssh mg-spark-slave-${i} \
    --command="cd /opt/spark && SPARK_WORKER_INSTANCES=4 ./sbin/start-slave.sh -c 16 -m 103G spark://mg-spark-master:7077"
done
