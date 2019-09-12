#!/bin/bash
##
#
# Script to build xgboost jar files.
# Source tree is supposed to be ready by Jenkins before starting this script.
#
###
set -e

BUILD_ARG="$1" 
if [ "$REPO_TYPE" == "Local" ];then
    BUILD_ARG="$BUILD_ARG -s jenkins/settings.xml -P apt-sh04-repo"
elif [ "$REPO_TYPE" == "Sonatype" ];then
    BUILD_ARG="$BUILD_ARG -P sonatype-repo"
else
   echo "Dependency gpuwa"
fi

# set .m2 dir and force update snapshot dependencies
BUILD_ARG="$BUILD_ARG -Dmaven.repo.local=$WORKSPACE/.m2 -U"
echo "mvn package $BUILD_ARG"

cd xgboost
mvn $BUILD_ARG clean package
cd -
