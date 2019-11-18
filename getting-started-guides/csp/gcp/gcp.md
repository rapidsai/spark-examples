# Get Started with XGBoost4J-Spark on GCP

This is a getting started guide to XGBoost4J-Spark on [Google Cloud Dataproc](https://cloud.google.com/dataproc). At the end of this guide, readers will be able to run a sample Spark RAPIDS XGBoost application on NVIDIA GPUs hosted by Google Cloud.


Prerequisites
-------------
* Apache Spark 2.3+
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better (V100, P100, T4 and later)
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * NVIDIA driver 410.48+
  * CUDA V10.1/10.0/9.2
  * NCCL 2.4.7 and later
* `EXCLUSIVE_PROCESS` must be set for all GPUs in each NodeManager.(Initialization script provided in this guide will set this mode by default)
* `spark.dynamicAllocation.enabled` must be set to False for spark


Before you begin, please make sure you have installed [Google Cloud SDK](https://cloud.google.com/sdk/) and selected your project directory on your local machine. The following steps require a GCP project directory and Google Storage bucket associated with the project directory.
There are three steps to run a sample PySpark XGBoost app using Jupyter notebook on a GCP GPU Cluster from your local machine. 
1. Initialization steps to download required files for Spark RAPIDS XGBoost app
2. Create a GPU Cluster with pre-installed GPU Drivers, Spark RAPIDS libraries, Spark XGBoost libraries and Jupyter notebook
3. Upload and run a sample XGBoost PySpark app to the Jupyter notebook on your GCP cluster. 
4. Optional step: Submit sample PySpark or Scala App using the gcloud command from your local machine


### Step 1.  Initialization steps to download required files for Spark RAPIDS XGBoost PySpark app

Before you create a cluster, please git clone the [spark-examples directory](https://github.com/rapidsai/spark-examples) to your local machine. `cd` into the spark-examples/getting-started-guides/csp/gcp/spark-gpu directory. Open the rapids.sh script using a text editor.  Modify the `GCS_BUCKET=my-bucket` line to specify your google GCP bucket name.  

Execute the commands below while in the spark-examples folder.  These commands will copy the following files into your GCP bucket: 

1. Initialization scripts for GPU and RAPIDS Spark, 
2. PySpark app files  
3. A sample dataset for a XGBoost PySpark app 
4. The latest Spark RAPIDS XGBoost jar files from the public maven repository

```
export GCS_BUCKET=my-bucket
pushd getting-started-guides/csp/gcp/spark-gpu
gsutil cp -r internal gs://$GCS_BUCKET/spark-gpu/
gsutil cp rapids.sh gs://$GCS_BUCKET/spark-gpu/rapids.sh
popd
pushd datasets/
tar -xvf mortgage-small.tar.gz
gsutil cp -r mortgage-small/ gs://$GCS_BUCKET/
popd
wget -O cudf-0.9.1-cuda10.jar https://search.maven.org/remotecontent?filepath=ai/rapids/cudf/0.9.1/cudf-0.9.1-cuda10.jar
wget -O xgboost4j_2.11-1.0.0-Beta2.jar https://search.maven.org/remotecontent?filepath=ai/rapids/xgboost4j_2.11/1.0.0-Beta2/xgboost4j_2.11-1.0.0-Beta2.jar
wget -O xgboost4j-spark_2.11-1.0.0-Beta2.jar https://search.maven.org/remotecontent?filepath=ai/rapids/xgboost4j-spark_2.11/1.0.0-Beta2/xgboost4j-spark_2.11-1.0.0-Beta2.jar
gsutil cp cudf-0.9.1-cuda10.jar xgboost4j-spark_2.11-1.0.0-Beta2.jar xgboost4j_2.11-1.0.0-Beta2.jar gs://$GCS_BUCKET/
````

After executing these commands, use your web browser to navigate to Google Cloud Platform console and make sure your Google storage bucket “my-bucket” directory structure has the following files:
* gs://my-bucket/spark-gpu/rapids.sh
* gs://my-bucket/spark-gpu/internal/install-gpu-driver-debian.sh
* gs://my-bucket/spark-gpu/internal/install-gpu-driver-ubuntu.sh
* gs://my-bucket/cudf-0.9.1-cuda10.jar
* gs://my-bucket/xgboost4j-spark_2.11-1.0.0-Beta2.jar
* gs://my-bucket/xgboost4j_2.11-1.0.0-Beta2.jar
* gs://my-bucket/mortgage-small/eval/mortgage-small.csv
* gs://my-bucket/mortgage-small/eval/mortgage-small.csv
* gs://my-bucket/mortgage-small/trainWithEval/test.csv


### Step 2. Create a GPU Cluster with pre-installed GPU drivers, Spark RAPIDS libraries, Spark XGBoost libraries and Jupyter Notebook 

Using the `gcloud` command creates a new cluster with Rapids Spark GPU initialization action. The following commands 
will create a new cluster named `<CLUSTER_NAME>` under your project directory. Here we use Ubuntu as our recommended 
OS for Spark-XGBoost on GCP. Modify the `GCS_BUCKET=my-bucket` line to specify your google GCP bucket name. Also 
modify `--properties` to include update-to-date jar file released by NVIDIA Spark XGBoost team. 

```
export CLUSTER_NAME=my-gpu-cluster
export ZONE=us-central1-b
export REGION=us-central1
export GCS_BUCKET=my-bucket
export INIT_ACTIONS_BUCKET=my-bucket
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
    --bucket $GCS_BUCKET \
    --metadata JUPYTER_PORT=8123,INIT_ACTIONS_REPO="gs://$INIT_ACTIONS_BUCKET",linux-dist="ubuntu",
    GCS_BUCKET="gs://$GCS_BUCKET" \
    --initialization-actions gs://$INIT_ACTIONS_BUCKET/spark-gpu/rapids.sh \
    --optional-components=ANACONDA,JUPYTER \
    --subnet=default \
    --properties '^#^spark:spark.dynamicAllocation.enabled=false#spark:spark.shuffle.service.enabled=false#spark:spark.submit.pyFiles=/usr/lib/spark/python/lib/xgboost4j-spark_2.11-1.0.0-Beta2.jar#spark:spark.jars=/usr/lib/spark/jars/xgboost4j-spark_2.11-1.0.0-Beta2.jar,/usr/lib/spark/jars/xgboost4j_2.11-1.0.0-Beta2.jar,/usr/lib/spark/jars/cudf-0.9.1-cuda10.jar' \
    --enable-component-gateway 
```

After submitting the commands, please go to the Google Cloud Platform console on your browser. Search for "Dataproc" and click on the "Dataproc" icon. This will navigate you to the Dataproc clusters page. “Dataproc” page lists all Dataproc clusters created under your project directory. You can see “my-gpu-cluster” with Status "Running". This cluster is now ready to host RAPIDS Spark XGBoost applications.  


### Step 3. Upload and run a sample XGBoost PySpark app to the Jupyter notebook on your GCP cluster.

To open the Jupyter notebook, click on the “my-gpu-cluster” under Dataproc page and navigate to the "Web Interfaces" Tab. Under the "Web Interfaces", click on the “Jupyter” link.
This will open the Jupyter Notebook. This notebook is running on the “my-gpu-cluster” we just created. 

Next, to upload the Sample PySpark App into the Jupyter notebook, use the “Upload” button on the Jupyter notebook. Sample Pyspark notebook is inside the `spark-examples/examples/notebooks/python/’ directory. Once you upload the sample mortgage-gpu.ipynb, make sure to change the kernel to “PySpark” under the "Kernel" tab using "Change Kernel" selection.The Spark XGBoost Sample Jupyter notebook is now ready to run on a “my-gpu-cluster”.
To run the Sample PySpark app on Jupyter notebook, please follow the instructions on the notebook and also update the data path for sample datasets.
```
train_data = GpuDataReader(spark).schema(schema).option('header', True).csv('gs://$GCS_BUCKET/mortgage-small/train')
eval_data = GpuDataReader(spark).schema(schema).option('header', True).csv('gs://$GCS_BUCKET/mortgage-small/eval')
```

### Step 4. [Optional] Submit Sample Apps 
#### 4a) Submit Scala Spark App on GPUs

Please build the `sample_xgboost_apps jar` with dependencies as specified in the [guide]
(/getting-started-guides/building-sample-apps/scala.md) and place the jar file (`sample_xgboost_apps-0.1.4-jar-with-dependencies.jar`) under the `gs://$GCS_BUCKET/spark-gpu` folder. To do this you can either drag and drop files from your local machine into the GCP [storage browser](https://console.cloud.google.com/storage/browser/rapidsai-test-1/?project=nv-ai-infra&organizationId=210881545417), or use the [gsutil cp](https://cloud.google.com/storage/docs/gsutil/commands/cp) as shown before to do this from a command line.

Use the following commands to submit sample Scala app on this GPU cluster.

```export MAIN_CLASS=ai.rapids.spark.examples.mortgage.GPUMain
    export RAPIDS_JARS=gs://$GCS_BUCKET/spark-gpu/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar
    export DATA_PATH=gs://$GCS_BUCKET
    export TREE_METHOD=gpu_hist
    export SPARK_NUM_EXECUTORS=4
    export CLUSTER_NAME=my-gpu-cluster
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
        -trainDataPath=${DATA_PATH}/mortgage-small/train/mortgage_small.csv \
        -evalDataPath=${DATA_PATH}/mortgage-small/eval/mortgage_small.csv \
        -maxDepth=8  
```

#### 4b) Submit  PySpark App on GPUs

Please build the sample_xgboost pyspark app as specified in the [guide](/getting-started-guides/building-sample-apps/python.md) and place the sample.zip file into GCP storage bucket.

Use the following commands to submit sample PySpark app on this GPU cluster.


```
    export DATA_PATH=gs://$GCS_BUCKET
    export LIBS_PATH=gs://$GCS_BUCKET
    export SPARK_DEPLOY_MODE=cluster
    export SPARK_PYTHON_ENTRYPOINT=${LIBS_PATH}/main.py
    export MAIN_CLASS=ai.rapids.spark.examples.mortgage.gpu_main
    export RAPIDS_JARS=${LIBS_PATH}/cudf-0.9.1-cuda10.jar,${LIBS_PATH}/xgboost4j_2.11-1.0.0-Beta2.jar,${LIBS_PATH}/xgboost4j-spark_2.11-1.0.0-Beta2.jar
    export SPARK_PY_FILES=${LIBS_PATH}/xgboost4j-spark_2.11-1.0.0-Beta2.jar,${LIBS_PATH}/sample.zip
    export TREE_METHOD=gpu_hist
    export SPARK_NUM_EXECUTORS=4
    export CLUSTER_NAME=my-gpu-cluster
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
        --trainDataPath=${DATA_PATH}/mortgage-small/train/mortgage_small.csv \
        --evalDataPath=${DATA_PATH}/mortgage-small/eval/mortgage_small.csv \
        --maxDepth=8 \
        --mainClass=${MAIN_CLASS}
```

#### 4c) Addendum: Submit a Spark Job on CPUs 

Submitting a CPU job on this cluster is very similar. Below's an example command that runs the same Mortgage application on CPUs:

```
    export GCS_BUCKET=my-bucket
    export MAIN_CLASS=ai.rapids.spark.examples.mortgage.CPUMain
    export RAPIDS_JARS=gs://$GCS_BUCKET/spark-gpu/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar
    export DATA_PATH=gs://$GCS_BUCKET
    export TREE_METHOD=hist
    export SPARK_NUM_EXECUTORS=4
    export CLUSTER_NAME=my-gpu-cluster
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
        -trainDataPath=${DATA_PATH}/mortgage-small/train/mortgage_small.csv \
        -evalDataPath=${DATA_PATH}/mortgage-small/eval/mortgage_small.csv \
        -maxDepth=8
```

### Step 5. Clean Up

When you're done working on this cluster, don't forget to delete the cluster, using the following command (replacing the highlighted cluster name with yours):

```bash
    gcloud dataproc clusters delete my-gpu-cluster
```

<sup>*</sup> Please see our release [announcement](https://news.developer.nvidia.com/gpu-accelerated-spark-xgboost/) for official performance benchmarks.
