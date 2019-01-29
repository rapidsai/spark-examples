#!/usr/bin/env bash

if [[ "$1" = "" ]]
then
  echo "Usage: $0 <Kubernetes master URL>"
  exit 1
fi

set -ex

/opt/spark/bin/spark-submit \
    --master k8s://$1 \
    --deploy-mode cluster \
    --name mortgage-etl \
    --class ai.rapids.sparkexamples.mortgage.ETL \
    --conf spark.executor.instances=2 \
    --conf spark.executor.cores=5 \
    --conf spark.kubernetes.container.image=rongou/spark-examples:latest \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.container.image.pullPolicy=Always \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.driver.volumes.hostPath.data.mount.path=/data \
    --conf spark.kubernetes.driver.volumes.hostPath.data.mount.readOnly=true \
    --conf spark.kubernetes.driver.volumes.hostPath.data.options.path=/hdd/mortgage \
    --conf spark.kubernetes.driver.volumes.hostPath.output.mount.path=/output \
    --conf spark.kubernetes.driver.volumes.hostPath.output.options.path=/hdd/spark \
    --conf spark.kubernetes.executor.volumes.hostPath.data.mount.path=/data \
    --conf spark.kubernetes.executor.volumes.hostPath.data.mount.readOnly=true \
    --conf spark.kubernetes.executor.volumes.hostPath.data.options.path=/hdd/mortgage \
    --conf spark.kubernetes.executor.volumes.hostPath.output.mount.path=/output \
    --conf spark.kubernetes.executor.volumes.hostPath.output.options.path=/hdd/spark \
    local:///opt/spark/examples/jars/mortgage-assembly-0.1.0-SNAPSHOT.jar \
    /data/perf/Performance_2007Q4.txt \
    /data/acq/Acquisition_2007Q4.txt \
    /output/pq
