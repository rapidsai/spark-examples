# Get Started with XGBoost4J-Spark on GCP

This is a getting started guide to XGBoost4J-Spark on [Google Cloud Dataproc](https://cloud.google.com/dataproc). At the end of this guide, the reader will be able to run a sample Apache Spark application that runs on NVIDIA GPUs on Google Cloud.

### Install & Cluster Spin up

Prerequisites
-------------
* Apache Spark 2.3+
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better (V100, P100, T4 and later)
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * NVIDIA driver 410.48+
  * CUDA V10.0/9.2
  * NCCL 2.4.7 and later
* `EXCLUSIVE_PROCESS` must be set for all GPUs in each NodeManager.
* `spark.dynamicAllocation.enabled` must be set to False for spark


There are four steps to run a Sample XGBoost4j app on a GCP GPU Cluster:
1. Create a GPU Cluster with preinstalled GPU Drivers, Spark RAPIDS libraries and Spark XGBoost libraries
2. Copy sample data in to GCS Bucket 
3. Prepare sample XGBoost apps on your local machine. 
4. Submit sample PySpark or Scala App using gcloud command from your local machine


#### Step 1.  Create a GPU Cluster in GCP

Before you create a cluster, please gitclone the spark-examples directory to your local machine. From the spark-examples directory, copy the spark-gpu initialization scripts inside `spark-gpu` folder into an accessible GCS bucket with following structure:

```bash
    /$STORAGE_BUCKET/spark-gpu/rapids.sh
    /$STORAGE_BUCKET/spark-gpu/internal/install-gpu-driver-ubuntu.sh
    /$STORAGE_BUCKET/spark-gpu/internal/install-gpu-driver-debian.sh
```  

Using the `gcloud` command create a new cluster with Rapids Spark GPU initialization action. The following command will create a new cluster named `<CLUSTER_NAME>` under your Project space. Here we use Ubuntu since that is our recommended OS for Spark-XGBoost on GCP.

```bash
    export CLUSTER_NAME=sparkgpu
    export ZONE=us-central1-b
    export REGION=us-central1
    export STORAGE_BUCKET=dataproc-initialization-actions
    export NUM_GPUS=2
    export NUM_WORKERS=2
    
    gcloud beta dataproc clusters create $CLUSTER_NAME  \
        --zone $ZONE \
        --region $REGION \
        --master-machine-type n1-standard-32 \
        --master-boot-disk-size 50 \
        --worker-accelerator type=nvidia-tesla-t4,count=$NUM_GPUS \
        --worker-machine-type n1-standard-32 \
        --worker-boot-disk-size 50 \
        --num-worker-local-ssds 1 \
        --num-workers $NUM_WORKERS \
        --image-version 1.4-ubuntu18 \
        --bucket $STORAGE_BUCKET \
        --metadata zeppelin-port=8081,INIT_ACTIONS_REPO="gs://$STORAGE_BUCKET",linux-dist="ubuntu" \
        --initialization-actions gs://$STORAGE_BUCKET/spark-gpu/rapids.sh \
        --optional-components=ZEPPELIN \
        --subnet=default \
        --properties 'spark:spark.dynamicAllocation.enabled=false,spark:spark.shuffle.service.enabled=false' \
        --enable-component-gateway
```

This cluster is now ready to host RAPIDS Spark XGBoost Applications.

#### Step 2. Copy sample data in to GCS Bucket 
Copy the sample data<link> into the same GCS bucket where the GPU initialization scripts are stored. The GCS directory structure should look like the followinf

```bash
    /$STORAGE_BUCKET/test/mortgage_eval_merged.csv
    /$STORAGE_BUCKET/train/mortgage_train_merged.csv
    /$STORAGE_BUCKET/spark-gpu/rapids.sh
    /$STORAGE_BUCKET/spark-gpu/internal/install-gpu-driver-ubuntu.sh
    /$STORAGE_BUCKET/spark-gpu/internal/install-gpu-driver-debian.sh
``` 

#### Step 3. Build Sample Apps
a) Scala App: Please build the sample_xgboost_apps jar with dependencies as specified in the [guide](/getting-started-guides/building-sample-apps/scala.md) and place the Jar into GCP storage bucket.

You can either drag and drop files from the GCP [storage browser](https://console.cloud.google.com/storage/browser/rapidsai-test-1/?project=nv-ai-infra&organizationId=210881545417), or use the [gsutil cp](https://cloud.google.com/storage/docs/gsutil/commands/cp) to do this from the command line.

b) PySpark App: Please build the sample_xgboost pyspark app as specified in the [guide](/getting-started-guides/building-sample-apps/python.md) and place the files into GCP storage bucket.

You can download the latest RAPIDS jars to your local machine using the following commands:

```bash
wget -O cudf-0.9.1-cuda10.jar https://search.maven.org/remotecontent?filepath=ai/rapids/cudf/0.9.1/cudf-0.9.1-cuda10.jar
wget -O xgboost4j_2.11-1.0.0-Beta2.jar https://search.maven.org/remotecontent?filepath=ai/rapids/xgboost4j_2.11/1.0.0-Beta2/xgboost4j_2.11-1.0.0-Beta2.jar
wget -O xgboost4j-spark_2.11-1.0.0-Beta2.jar https://search.maven.org/remotecontent?filepath=ai/rapids/xgboost4j-spark_2.11/1.0.0-Beta2/xgboost4j-spark_2.11-1.0.0-Beta2.jar
```

Make sure you copy all the downloaded jar files into same GCS bucket. The directory structure should look like the following(if you built both Scala and PySpark Apps:
```bash
    /$STORAGE_BUCKET/test/mortgage_eval_merged.csv
    /$STORAGE_BUCKET/train/mortgage_train_merged.csv
    /$STORAGE_BUCKET/spark-gpu/rapids.sh
    /$STORAGE_BUCKET/spark-gpu/internal/install-gpu-driver-ubuntu.sh
    /$STORAGE_BUCKET/spark-gpu/internal/install-gpu-driver-debian.sh
    /$STORAGE_BUCKET/spark-gpu/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar
    /$STORAGE_BUCKET/spark-gpu/cudf-0.9.1-cuda10.jar
    /$STORAGE_BUCKET/xgboost4j_2.11-1.0.0-Beta2.jar
    /$STORAGE_BUCKET/xgboost4j-spark_2.11-1.0.0-Beta2.jar
    /$STORAGE_BUCKET/main.py
    /$STORAGE_BUCKET/sample.zip
``` 


#### Step 4. Submitting Jobs

### Use the following command to submit sample Scala app on this GPU cluster.

```bash
    export STORAGE_BUCKET=dataproc-initialization-actions
    export MAIN_CLASS=ai.rapids.spark.examples.mortgage.GPUMain
    export RAPIDS_JARS=gs://$STORAGE_BUCKET/spark-gpu/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar
    export DATA_PATH=$STORAGE_BUCKET
    export TREE_METHOD=gpu_hist
    export SPARK_NUM_EXECUTORS=4
    export CLUSTER_NAME=sparkgpu
    export REGION=us-central1

    gcloud beta dataproc jobs submit spark \
        --cluster=$CLUSTER_NAME \
        --region=$REGION \
        --class=$MAIN_CLASS \
        --jars=$RAPIDS_JARS \
        --properties=spark.executor.cores=1,spark.executor.instances=${SPARK_NUM_EXECUTORS},spark.executor.memory=8G,spark.executorEnv.LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:/usr/local/cuda-10.0/lib64:${LD_LIBRARY_PATH} \
        -- \
        -format=csv \
        -numRound=100 \
        -numWorkers=${SPARK_NUM_EXECUTORS} \
        -treeMethod=${TREE_METHOD} \
        -trainDataPath=${DATA_PATH}/train/mortgage_train_merged.csv \
         -evalDataPath=${DATA_PATH}/test/mortgage_eval_merged.csv \
         -maxDepth=8  
```

Here's a quick explanation of key parameters:

- CLUSTER_NAME: name of the cluster you had created in the first step
- MAIN_CLASS: Class containing the main method to run
- RAPIDS_JARS: All the jar files you file need. This includes the RAPIDS jars as well as the one for the application you are submitting.
properties:  Use this to specify Spark properties. The command above includes the ones you likely need.

You can check out the full documentation of this api [here](https://cloud.google.com/sdk/gcloud/reference/beta/dataproc/jobs/submit/spark).


### Use the following command to submit sample PySpark app on this GPU cluster.

```bash
    export DATA_PATH=gs://$STORAGE_BUCKET
    export LIBS_PATH=gs://$STORAGE_BUCKET
    export SPARK_DEPLOY_MODE=cluster
    export SPARK_PYTHON_ENTRYPOINT=${LIBS_PATH}/main.py
    export MAIN_CLASS=ai.rapids.spark.examples.mortgage.gpu_main
    export RAPIDS_JARS=${LIBS_PATH}/cudf-0.9.1-cuda10.jar,${LIBS_PATH}/xgboost4j_2.11-1.0.0-Beta2.jar,${LIBS_PATH}/xgboost4j-spark_2.11-1.0.0-Beta2.jar
    export SPARK_PY_FILES=${LIBS_PATH}/xgboost4j-spark_2.11-1.0.0-Beta2.jar,${LIBS_PATH}/sample.zip
    export TREE_METHOD=gpu_hist
    export SPARK_NUM_EXECUTORS=4
    export CLUSTER_NAME=sparkgputest2
    export REGION=us-central1

gcloud beta dataproc jobs submit pyspark \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --jars=$RAPIDS_JARS \
    --properties=spark.executor.cores=1,spark.executor.instances=${SPARK_NUM_EXECUTORS},spark.executor.memory=8G,spark.executorEnv.LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:/usr/local/cuda-10.0/lib64:${LD_LIBRARY_PATH} \
    --py-files=${SPARK_PY_FILES} \
    ${SPARK_PYTHON_ENTRYPOINT} \
    -- \
    --format=csv \
    --numRound=100 \
    --numWorkers=${SPARK_NUM_EXECUTORS} \
    --treeMethod=${TREE_METHOD} \
    --trainDataPath=${DATA_PATH}/train/mortgage_train_merged.csv \
    --evalDataPath=${DATA_PATH}/test/mortgage_eval_merged.csv \
    --maxDepth=8 \
    --mainClass=${MAIN_CLASS}
```


#### Addendum: Submit Spark Job on CPUs 

Submitting a CPU job on this cluster is very similar. Given below iss an example command that runs the same Mortgage application on CPUs:

```bash
    export STORAGE_BUCKET=dataproc-initialization-actions
    export MAIN_CLASS=ai.rapids.spark.examples.mortgage.CPUMain
    export RAPIDS_JARS=gs://$STORAGE_BUCKET/spark-gpu/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar
    export DATA_PATH=hdfs:///tmp/xgboost4j_spark/mortgage/csv
    export TREE_METHOD=hist
    export SPARK_NUM_EXECUTORS=4
    export CLUSTER_NAME=sparkgpu
    export REGION=us-central1

    gcloud beta dataproc jobs submit spark \
        --cluster=$CLUSTER_NAME \
        --region=$REGION \
        --class=$MAIN_CLASS \
        --jars=$RAPIDS_JARS \
        --properties=spark.executor.cores=1,spark.executor.instances=${SPARK_NUM_EXECUTORS},spark.executor.memory=8G,spark.executorEnv.LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:/usr/local/cuda-10.0/lib64:${LD_LIBRARY_PATH} \
        -- \
        -format=csv \
        -numRound=100 \
        -numWorkers=${SPARK_NUM_EXECUTORS} \
        -treeMethod=${TREE_METHOD} \
        -trainDataPath=${DATA_PATH}/train/mortgage_train_merged.csv \
         -evalDataPath=${DATA_PATH}/test/mortgage_eval_merged.csv \
         -maxDepth=8
```

### Clean Up

When you're done working on this cluster, don't forget to delete the cluster, using the following command (replacing the highlighted cluster name with yours):

```bash
    gcloud dataproc clusters delete $CLUSTER_NAME
```

<sup>*</sup> Please see our release announcement for official performance benchmarks.
