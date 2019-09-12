#!/bin/bash
##
#
# Script to run test app for xgboost jar files in spark 2.3.3.
#
###
set -e

CLASSIFIERS=""
#Set defaut cuda9 as default
CUDA_VER=$1
if [ "$CUDA_VER" == "10.0" ]; then
    CLASSIFIERS="-cuda10"
else
    CLASSIFIERS=""
fi

echo "CUDA_VER: $CUDA_VER, CLASSIFIERS: $CLASSIFIERS"

#Get cudf version, default: "0.8-SNAPSHOT"
CUDF_VER=$2
if [ "CUDF_VER"x == x ];then
    CUDF_VER="0.9-SNAPSHOT"
fi

#Get xgboost version, default: "0.90-SNAPSHOT"
XGB_VER=$3
if [ "XGB_VER"x == x ];then
    XGB_VER="1.0.0-SNAPSHOT"
fi

#Set test data path
DATA_PATH=/data
SIMPLE_DATA=$4
if [[ $SIMPLE_DATA == "true" ]] && [[ -d /data/simple_data ]];then
DATA_PATH=/data/simple_data
fi

#Set default spark-2.4.3
SPARK_VER=$5
if [ "$SPARK_VER" != "2.3.3" ];then
    SPARK_VER=2.4.3
fi

#Get example name & version, default: sample_xgboost_apps 0.1.4
cd xgboost
#EXAMPLE_NAME=`mvn exec:exec -q --non-recursive -Dexec.executable=echo -Dexec.args='${project.name}'`
if [ "$EXAMPLE_NAME"x == x ];then
    EXAMPLE_NAME="sample_xgboost_apps"
fi
#EXAMPLE_VER=`mvn exec:exec -q --non-recursive -Dexec.executable=echo -Dexec.args='${project.version}'`
if [ "$EXAMPLE_VER"x == x ];then
    EXAMPLE_VER="0.1.4"
fi
echo "EXAMPLE_NAME: $EXAMPLE_NAME, EXAMPLE_VER: $EXAMPLE_VER"
cd -

#Set jar file path
M2_REPO_RAPIDS="$WORKSPACE/.m2/ai/rapids"
CUDF_JAR_FILE="$M2_REPO_RAPIDS/cudf/$CUDF_VER/cudf-$CUDF_VER$CLASSIFIERS.jar"
XGBOOST4J_JAR_FILE="$M2_REPO_RAPIDS/xgboost4j_2.11/$XGB_VER/xgboost4j_2.11-$XGB_VER.jar"
XGBOOST4J_SPARK_JAR_FILE="$M2_REPO_RAPIDS/xgboost4j-spark_2.11/$XGB_VER/xgboost4j-spark_2.11-$XGB_VER.jar"
SAMPLE_APP_JAR_FILE="$WORKSPACE/xgboost/target/$EXAMPLE_NAME-$EXAMPLE_VER.jar"
SPARK_CUDA_JARS="$CUDF_JAR_FILE,$XGBOOST4J_JAR_FILE,$XGBOOST4J_SPARK_JAR_FILE $SAMPLE_APP_JAR_FILE"

AGARICUS_CSV_TRAIN_PATH=$DATA_PATH/agaricus/csv/train
AGARICUS_CSV_TEST_PATH=$DATA_PATH/agaricus/csv/test

MORTGAGE_CSV_TRAIN_PATH=$DATA_PATH/mortgage/csv/train/2008Q2
MORTGAGE_CSV_TEST_PATH=$DATA_PATH/mortgage/csv/eval/2008Q2
MORTGAGE_PARQUET_TRAIN_PATH=$DATA_PATH/mortgage/parquet/train/2008Q2
MORTGAGE_PARQUET_TEST_PATH=$DATA_PATH/mortgage/parquet/eval/2008Q2
MORTGAGE_ORC_TRAIN_PATH=$DATA_PATH/mortgage/orc/train/2008Q2
MORTGAGE_ORC_TEST_PATH=$DATA_PATH/mortgage/orc/eval/2008Q2

TAXI_CSV_TRAIN_PATH=$DATA_PATH/taxi/csv/train/2009-01-03
TAXI_CSV_TEST_PATH=$DATA_PATH/taxi/csv/test/2009-01-03
TAXI_PARQUET_TRAIN_PATH=$DATA_PATH/taxi/parquet/train/2009-01-03
TAXI_PARQUET_TEST_PATH=$DATA_PATH/taxi/parquet/test/2009-01-03
TAXI_ORC_TRAIN_PATH=$DATA_PATH/taxi/orc/train/2009-01-03
TAXI_ORC_TEST_PATH=$DATA_PATH/taxi/orc/test/2009-01-03

#---------- Set environment ---------
SPARK_HOME="/usr/local/spark-$SPARK_VER-bin-hadoop2.7"
PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
echo "set path=$PATH and spark home=$SPARK_HOME"

#-----------------------------------------------------------------------------------------
echo "----------------------------------- Run testing --------------------------------------"
# Start testing  ...
echo "Start testing ..."
stop-spark.sh
start-spark.sh
jps

#Set spark args and xgboost args
export XGBOOST_ARGS=""
export SPARK_SUBMIT_ARGS=""
#$0:data source(agaricus/mortgate/taxi) $1:on device(cpu/gpu) $2: data format(csv/parquet/orc)
get_xgboost_args() {
    typeset -u ARG_DATA
    typeset -u ARG_XPU
    typeset -u ARG_FORMAT
    ARG_DATA=$1
    ARG_XPU=$2
    ARG_FORMAT=$3
    echo $ARG_DATA $ARG_XPU $ARG_FORMAT

    TRAIN_PATH="$ARG_DATA"_"$ARG_FORMAT"_"TRAIN_PATH"
    TEST_PATH="$ARG_DATA"_"$ARG_FORMAT"_"TEST_PATH"

    SPARK_SUBMIT_ARGS="--class ai.rapids.spark.examples."$1"."$ARG_XPU"Main \
        --master spark://$HOSTNAME:7077 --executor-memory 32G --jars $SPARK_CUDA_JARS"

    XGBOOST_ARGS="-numRound=100 -overwrite=true -format=$3 \
        -modelPath=/tmp/models/$1/$2 \
        -trainDataPath=${!TRAIN_PATH} -evalDataPath=${!TEST_PATH}"

    if [ $2 == "cpu" ];then
        XGBOOST_ARGS="-numWorkers=6 -treeMethod=hist "$XGBOOST_ARGS
    fi
}

# Start test
get_xgboost_args "agaricus" "cpu" "csv"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

get_xgboost_args "agaricus" "gpu" "csv"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

get_xgboost_args "mortgage" "cpu" "csv"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

get_xgboost_args "mortgage" "gpu" "csv"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

get_xgboost_args "taxi" "cpu" "csv"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

get_xgboost_args "taxi" "gpu" "csv"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

echo " ----------------------------- parquet data: ---------------------------------------"
get_xgboost_args "mortgage" "cpu" "parquet"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

get_xgboost_args "mortgage" "gpu" "parquet"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

get_xgboost_args "taxi" "cpu" "parquet"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

get_xgboost_args "taxi" "gpu" "parquet"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

echo " ----------------------------- ORC data: ---------------------------------------"
get_xgboost_args "mortgage" "cpu" "orc"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

get_xgboost_args "mortgage" "gpu" "orc"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

get_xgboost_args "taxi" "cpu" "orc"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

get_xgboost_args "taxi" "gpu" "orc"
spark-submit $SPARK_SUBMIT_ARGS $XGBOOST_ARGS

echo "----------------------------------- FINISHING --------------------------------------"
