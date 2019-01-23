# Spark Examples

This repository contains examples for running Spark on GPUs, leveraging [RAPIDS](https://rapids.ai).

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

Install CUDA 10:
```bash
wget -O cuda_10.0.130_410.48_linux.run https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
sudo sh cuda_10.0.130_410.48_linux.run
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
sudo apt install openjdk-8-jdk maven
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update
sudo apt-get install sbt
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
sbt aseembly
```

### Running on Google Cloud Platform (GCP)

Assuming you have a Google Compute Engine (GCE) template instance satisfying the following requirements:
*   `n1-highmem-16` with 1 x NVIDIA Tesla T4
*   Ubuntu 18.04
*   NVIDIA driver >= 410.48
*   CUDA 10.0 (10.0.130)
*   NCCL 2.3.7-1
*   Java 1.8.0
*   Python 2.7 (for XGBoost tracker)
*   Spark 2.4.0 (installed under `/opt/spark`)
*   An NFS volume mounted under `/data`, with the mortgage data under `/data/mortgage`

Create instances for the Spark cluster:
```bash
source deploy/gcp/instances.sh
export INSTANCE_TEMPLATE=spark-1xt4
gcloud compute instances create $INSTANCES --source-instance-template $INSTANCE_TEMPLATE --async
``` 

Start the Spark cluster in standalone mode:
```bash
./deploy/gcp/start_cluster.sh
```

Run the Spark job:
```bash
# Comment/uncomment lines to run the specific parameter combination:
./deploy/gcp/run_job.sh
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
gcloud compute instances stop $INSTANCES --async
```

or delete them:
```bash
source deploy/gcp/instances.sh
gcloud compute instances delete $INSTANCES --async
```
