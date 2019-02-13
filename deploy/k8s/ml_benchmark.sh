#!/usr/bin/env bash

# Override on the command line to use a different docker image.
: ${DOCKER_IMAGE:=rongou/spark-examples:latest}

# Override on the command line to use a different output directory.
: ${OUTPUT_DIR:=/hdd/spark}

if [[ "$1" = "" ]]
then
  echo "Usage: $0 <Kubernetes master URL>"
  exit 1
fi

set -ex

/opt/spark/bin/spark-submit \
    --master k8s://$1 \
    --deploy-mode cluster \
    --name mortgage-ml-benchmark \
    --class ai.rapids.sparkexamples.mortgage.MLBenchmark \
    --conf spark.executor.instances=2 \
    --executor-memory 13G \
    --conf spark.kubernetes.executor.podTemplateFile=deploy/k8s/executor-template.yaml \
    --conf spark.kubernetes.container.image=${DOCKER_IMAGE} \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.container.image.pullPolicy=Always \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.driver.volumes.hostPath.output.mount.path=/output \
    --conf spark.kubernetes.driver.volumes.hostPath.output.options.path=${OUTPUT_DIR} \
    --conf spark.kubernetes.executor.volumes.hostPath.output.mount.path=/output \
    --conf spark.kubernetes.executor.volumes.hostPath.output.options.path=${OUTPUT_DIR} \
    --conf spark.executorEnv.NCCL_DEBUG=INFO \
    local:///opt/spark/examples/jars/mortgage-assembly-0.1.0-SNAPSHOT.jar \
    /output/pq \
    /output/benchmark/2007Q4-gpu-8 \
    2 \
    1 \
    100 \
    1 \
    gpu_hist \
    8 \
    depthwise
