# Get Started with XGBoost4J-Spark on AWS EMR

This is a getting started guide for XGBoost4J-Spark on AWS EMR. At the end of this guide, the user will be able to run a sample Apache Spark application that runs on NVIDIA GPUs on AWS EMR.

For more information on AWS EMR, please see the [AWS documentation](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-what-is-emr.html).

### Configure and Launch AWS EMR with GPU Nodes

Go to the AWS Management Console and select the `EMR` service from the "Analytics" section. Choose the region you want to launch your cluster in, e.g. US West Oregon, using the dropdown menu in the top right corner. Click `Create cluster` and select `Go to advanced options`, which will bring up a detailed cluster configuration page.

##### Step 1:  Software and Steps

Select **emr-5.27.0** for the release, uncheck all the software options, and then check **Hadoop 2.8.5** and **Spark 2.4.4**.  (Any EMR version that supports Spark 2.3 or above will work).

In the "Edit software settings" field, add the following snippet to disable Spark Dynamic Allocation by default: `[{"classification":"spark-defaults","properties":{"spark.dynamicAllocation.enabled":"false"}}]`

![Step 1: Software and Steps](pics/emr-step-one-software-and-steps.png)

##### Step 2: Hardware

Select the desired VPC and availability zone in the "Network" and "EC2 Subnet" fields respectively. (Default network and subnet are ok)

In the "Core" node row, change the "Instance type" to **p3.2xlarge** and ensure "Instance count" is set to **2**. Keep the default "Master" node instance type of **m3.xlarge** and ignore the unnecessary "Task" node configuration.

![Step 2: Hardware](pics/emr-step-two-hardware.png)

##### Step 3:  General Cluster Settings

Enter a custom "Cluster name" and make a note of the s3 folder that cluster logs will be written to.

*Optionally* add key-value "Tags", configure a "Custom AMI", or add custom "Bootstrap Actions"  for the EMR cluster on this page.

![Step 3: General Cluster Settings](pics/emr-step-three-general-cluster-settings.png)

#####  Step 4: Security

Select an existing "EC2 key pair" that will be used to authenticate SSH access to the cluster's nodes. If you do not have access to an EC2 key pair, follow these instructions to [create an EC2 key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair).

*Optionally* set custom security groups in the "EC2 security groups" tab.

In the "EC2 security groups" tab, confirm that the security group chosen for the "Master" node allows for SSH access. Follow these instructions to [allow inbound SSH traffic](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/authorizing-access-to-an-instance.html) if the security group does not allow it yet.

![Step 4: Security](pics/emr-step-four-security.png)

##### Finish Cluster Configuration

The EMR cluster management page displays the status of multiple clusters or detailed information about a chosen cluster. In the detailed cluster view, the "Summary" and "Hardware" tabs can be used to monitor the status of master and core nodes as they provision and initialize.

![Cluster Details](pics/emr-cluster-details.png )

When the cluster is ready, a green-dot will appear next to the cluster name and the "Status" column will display **Waiting, cluster ready**.

![Cluster Waiting](pics/emr-cluster-waiting.png)

In the cluster's "Summary" tab, find the "Master public DNS" field and click the `SSH` button. Follow the instructions to SSH to the new cluster's master node.

![Cluster DNS](pics/emr-cluster-dns.png)

![Cluster SSH](pics/emr-cluster-ssh.png)

#####  Above Cluster can also be built using AWS CLI

```
aws emr create-cluster --termination-protected --applications Name=Hadoop Name=Spark Name=Zeppelin Name=Livy --tags 'Name=nvidia-gpu-spark' --ec2-attributes '{"KeyName":"your-key-name","InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"your-subnet-ID","EmrManagedSlaveSecurityGroup":"your-EMR-slave-security-group-ID","EmrManagedMasterSecurityGroup":"your-EMR-master-security-group-ID"}' --release-label emr-5.27.0 --log-uri 's3n://aws-logs-354625738399-us-west-2/elasticmapreduce/' --instance-groups '[{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"MASTER","InstanceType":"m5.xlarge","Name":"Master - 1"},{"InstanceCount":2,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":4}]},"InstanceGroupType":"CORE","InstanceType":"p3.2xlarge","Name":"Core - 2"}]' --configurations '[{"Classification":"spark-defaults","Properties":{"spark.dynamicAllocation.enabled":"false"}}]' --auto-scaling-role EMR_AutoScaling_DefaultRole --ebs-root-volume-size 10 --service-role EMR_DefaultRole --enable-debugging --name 'nvidia-gpu-spark' --scale-down-behavior TERMINATE_AT_TASK_COMPLETION --region us-west-2
```
Fill with actual value for KeyName, SubnetId, EmrManagedSlaveSecurityGroup, EmrManagedMasterSecurityGroup, name and region.


### Build and Execute XGBoost-Spark examples on EMR

SSH to the EMR cluster's master node and run the following steps to setup, build, and run the XGBoost-Spark examples.

#### Install git and maven

```
sudo yum update -y
sudo yum install git -y
wget http://apache.mirrors.lucidnetworks.net/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.zip
unzip apache-maven-3.6.3-bin.zip
export PATH=/home/hadoop/apache-maven-3.6.3/bin:$PATH
mvn --version
```

#### Build Example Jars

```
git clone https://github.com/rapidsai/spark-examples.git
pushd spark-examples/examples/apps/scala
mvn package #CUDA 9.2 build command
popd
```

The `mvn package` command may require additional configuration depending on the CUDA version of the chosen EMR instance images. For detailed build instructions including different CUDA versions, see [Build XGBoost Scala Examples](/getting-started-guides/building-sample-apps/scala.md).

#### Fetch the Mortgage Dataset

```
mkdir data
pushd data
wget https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip
unzip mortgage.zip
popd
```

#### Upload Data and Jar files to HDFS

```
hadoop fs -mkdir -p /tmp/xgboost4j_spark/data
hadoop fs -copyFromLocal ~/spark-examples/examples/apps/scala/target/*.jar /tmp/xgboost4j_spark
hadoop fs -copyFromLocal ~/data/* /tmp/xgboost4j_spark/data
```

#### Launch the GPU Mortgage Example

```
# location where data was downloaded
export DATA_PATH=hdfs:/tmp/xgboost4j_spark/data
# location for the required jar
export JARS_PATH=hdfs:/tmp/xgboost4j_spark
# spark deploy mode (see Apache Spark documentation for more information)
export SPARK_DEPLOY_MODE=cluster
# run a single executor for this example to limit the number of spark tasks and
# partitions to 1 as currently this number must match the number of input files
export SPARK_NUM_EXECUTORS=2
# spark driver memory
export SPARK_DRIVER_MEMORY=4g
# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g
# example class to use
export EXAMPLE_CLASS=ai.rapids.spark.examples.mortgage.GPUMain
# XGBoost4J example jar
export JAR_EXAMPLE=${JARS_PATH}/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar
# tree construction algorithm
export TREE_METHOD=gpu_hist


spark-submit                                                                    \
 --master yarn                                                                  \
 --deploy-mode ${SPARK_DEPLOY_MODE}                                             \
 --num-executors ${SPARK_NUM_EXECUTORS}                                         \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --class ${EXAMPLE_CLASS}                                                       \
 ${JAR_EXAMPLE}                                                                 \
 -trainDataPath=${DATA_PATH}/mortgage/csv/train/mortgage_train_merged.csv       \
 -evalDataPath=${DATA_PATH}/mortgage/csv/test/mortgage_eval_merged.csv          \
 -format=csv                                                                    \
 -numWorkers=${SPARK_NUM_EXECUTORS}                                             \
 -treeMethod=${TREE_METHOD}                                                     \
 -numRound=100                                                                  \
 -maxDepth=8
```

Retrieve the Spark driver's logs from the EMR cluster's "Application history" tab. Select the completed mortgage example's ID from the "Application ID" column and then select the "Executors" tab. In the **driver** row, click on `View logs` then `stdout`. The stdout log file contains time metrics and RMSE accuracy metrics.

![View Logs](pics/emr-view-logs.png)

![Stdout](pics/emr-stdout.png)
