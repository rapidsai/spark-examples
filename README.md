# Spark Examples

This repository contains examples for running Spark on GPUs, leveraging [RAPIDS](https://rapids.ai).

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Mortgage](#mortgage)
  - [Prerequisites](#prerequisites)
  - [Building XGBoost](#building-xgboost)
  - [Building the mortgage example](#building-the-mortgage-example)
  - [Running locally](#running-locally)
  - [Running on Google Cloud Platform (GCP)](#running-on-google-cloud-platform-gcp)
  - [Running on Google Cloud Platform (GCP) with multi-GPU VMs](#running-on-google-cloud-platform-gcp-with-multi-gpu-vms)
  - [Running on Kubernetes (K8S)](#running-on-kubernetes-k8s)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Mortgage

This example shows running XGBoost on the [mortgage data](https://rapidsai.github.io/demos/datasets/mortgage-data).

### Prerequisites

For now XGBoost needs to be built from source. Assuming you have access to a computer with an NVIDIA GPU running Ubuntu
18.04.

Install NVIDIA drivers:
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo ubuntu-drivers autoinstall
```

Install CUDA 10.1:
```bash
wget -q https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.168_418.67_linux.run
sudo sh cuda_10.1.168_418.67_linux.run
# Follow the command-line prompts, but don't install the driver again.
```

Install NCCL:
```bash
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt update
sudo apt install libnccl2 libnccl-dev
```

Install the Java toolchain:
```bash
sudo apt update
sudo apt install openjdk-8-jdk
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt update
sudo apt install sbt
```

Install maven:
```bash
export MAVEN_VERSION=3.6.1
wget -q https://www-us.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz
tar xzvf apache-maven-${MAVEN_VERSION}-bin.tar.gz -C /opt
ln -s /opt/apache-maven-${MAVEN_VERSION} /opt/maven
```
and add the following to your `.bashrc`:
```bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=/opt/maven/bin:${PATH}
```

Install cmake:
```bash
wget -q https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.tar.gz
sudo tar --strip-components=1 -xzvf cmake-3.14.4-Linux-x86_64.tar.gz -C /usr/local
```

Install additional packages for running XGBoost:
```bash
sudo apt update
sudo apt install python libgomp1

```

### Building XGBoost

From your root source directory (e.g. `${HOME}/src`), run:
```bash
git clone -b spark-gpu-example --recurse-submodules https://github.com/rongou/xgboost.git
cd xgboost/jvm-packages
mvn -DskipTests install
```

### Building the mortgage example

From your root source directory (e.g. `${HOME}/src`), run:
```bash
git clone https://github.com/rapidsai/spark-examples.git
cd spark-examples
sbt assembly
```

### Running locally

Download and install Spark:
```bash
wget -O spark-2.4.3-bin-hadoop2.7.tgz \
  "https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz"
tar xzvf spark-2.4.3-bin-hadoop2.7.tgz -C /opt
ln -s /opt/spark-2.4.3-bin-hadoop2.7 /opt/spark
```

Copy the mortgage example jar file:
```bash
cp ${HOME}/src/spark-examples/mortgage/target/scala-2.11/mortgage-assembly-0.1.0-SNAPSHOT.jar /opt/spark/examples/jars/
``` 

Download one year worth of the mortgage data (this uses 3.9 GB disk space):
```bash
wget http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_2000.tgz
mkdir -p /opt/data/mortgage
tar xzvf mortgage_2000.tgz -C /opt/data/mortgage
```

Start a Spark standalone cluster:
```bash
/opt/spark/sbin/start-master.sh
/opt/spark/sbin/start-slave.sh spark://${HOSTNAME}:7077
```
the Spark web UI can then be accessed at [http://localhost:8080](http://localhost:8080).

Run the ETL job (`executor-memory` should be adjusted depending on how much memory you have on your machine):
```bash
/opt/spark/bin/spark-submit \
  --class ai.rapids.sparkexamples.mortgage.ETL \
  --master spark://${HOSTNAME}:7077 \
  --deploy-mode cluster \
  --executor-memory 28G \
  /opt/spark/examples/jars/mortgage-assembly-0.1.0-SNAPSHOT.jar \
  /opt/data/mortgage/perf/Performance_2000Q1.txt \
  /opt/data/mortgage/acq/Acquisition_2000Q1.txt \
  /opt/models/mortgage/pq
```

Run the ML job:
```bash
WORKERS=1
SAMPLES=1
ROUNDS=100
THREADS=1
TREE_METHOD=gpu_hist
MAX_DEPTH=8
GROW_POLICY=depthwise
EXTERNAL_MEMORY=false
/opt/spark/bin/spark-submit \
  --class ai.rapids.sparkexamples.mortgage.MLBenchmark \
  --master spark://${HOSTNAME}:7077 \
  --deploy-mode cluster \
  --executor-memory 28G \
  --conf spark.executorEnv.NCCL_DEBUG=INFO \
  /opt/spark/examples/jars/mortgage-assembly-0.1.0-SNAPSHOT.jar \
  /opt/models/mortgage/pq \
  /opt/models/mortgage/benchmark/gpu-depth-8 \
  ${WORKERS} \
  ${SAMPLES} \
  ${ROUNDS} \
  ${THREADS} \
  ${TREE_METHOD} \
  ${MAX_DEPTH} \
  ${GROW_POLICY} \
  ${EXTERNAL_MEMORY}
```
benchmark results are written to `/opt/models/mortgage/benchmark`.

Finally, clean up Spark:
```bash
/opt/spark/sbin/stop-slave.sh
/opt/spark/sbin/stop-master.sh
```

### Running on Google Cloud Platform (GCP)

Assuming you have a Google Compute Engine (GCE) template instance satisfying the following requirements:
*   `n1-highmem-16` with 1 x NVIDIA Tesla T4
*   Ubuntu 18.04
*   NVIDIA driver >= 418.67
*   CUDA 10.1 Update 1 (10.1.168)
*   NCCL 2.4.7-1
*   Java 1.8.0
*   Python 2.7 (for XGBoost tracker)
*   Spark 2.4.3 (installed under `/opt/spark`)
*   An NFS volume mounted under `/data`, with the mortgage data under `/data/mortgage`

Create instances for the Spark cluster:
```bash
source deploy/gcp/instances.sh
export INSTANCE_TEMPLATE=spark-1xt4
gcloud compute instances create ${INSTANCES} --source-instance-template ${INSTANCE_TEMPLATE} --async
``` 

If the instances had been previously created but were stopped, start them again:
```bash
source deploy/gcp/instances.sh
gcloud compute instances start ${INSTANCES} --async
```

Start the Spark cluster in standalone mode:
```bash
./deploy/gcp/start_cluster.sh
```

Copy the jar file to NFS:
```bash
gcloud compute scp mortgage/target/scala-2.11/mortgage-assembly-0.1.0-SNAPSHOT.jar spark-master:/data/spark/jars/
```

Run the Spark job:
```bash
gcloud compute instances list --filter=name=spark-master # Note the external IP address
# Comment/uncomment lines to run the specific parameter combination:
./deploy/gcp/run_job.sh <Spark master IP>
```

While the job is running, access the Spark master at http://${SPARK_MASTER_IP}:8080.

To see the job details, set up an SSH tunnel:
```bash
gcloud compute ssh spark-master -- -N -p 22 -D localhost:5000
```
then set the `Network Proxy` to use `Socks Host` as `localhost` and port `5000`.

Performance numbers are written to NFS under `/data/spark/benchmark`:
```bash
gcloud compute ssh spark-master
cd /data/spark/benchmark
``` 

Finally, after finish running jobs, stop the instances:
```bash
source deploy/gcp/instances.sh
gcloud compute instances stop ${INSTANCES} --async
```

or delete them:
```bash
source deploy/gcp/instances.sh
gcloud compute instances delete ${INSTANCES}
```

### Running on Google Cloud Platform (GCP) with multi-GPU VMs

To run on GCP using multiple GPUs in each VM, you need `n1-highmem-64` instances with 4 x NVIDIA Tesla T4s. The steps
are similar.

Create instances for the Spark cluster:
```bash
source deploy/gcp/multigpu-instances.sh
export INSTANCE_TEMPLATE=spark-4xt4
gcloud compute instances create ${INSTANCES} --source-instance-template ${INSTANCE_TEMPLATE} --async
``` 

Start the Spark cluster in standalone mode:
```bash
./deploy/gcp/start_multigpu_cluster.sh
```

Then submit jobs as in the previous section.

### Running on Kubernetes (K8S)

Assuming you have a Kubernetes cluster with GPUs configured.

First, build and push the docker image:
```bash
docker build -t ${DOCKER_IMAGE} .
docker push ${DOCKER_IMAGE}
```

Then run the ETL job:
```bash
./deploy/k8s/etl.sh ${K8S_MASTER}
```

where the ${K8S_MASTER} can be obtained by running `kubectl cluster-info`.

Finally, run the ML job:
```bash
./deploy/k8s/ml_benchmark.sh ${K8S_MASTER}
```

Performance numbers are written under `${OUTPUT_DIR}/benchmark`.
