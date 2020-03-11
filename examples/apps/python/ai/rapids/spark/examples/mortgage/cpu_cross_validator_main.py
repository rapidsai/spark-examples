#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from ai.rapids.spark.examples.mortgage.consts import *
from ai.rapids.spark.examples.utility.utils import *
from ml.dmlc.xgboost4j.scala.spark import *
from ml.dmlc.xgboost4j.scala.spark.rapids import CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import SparkSession

def main(args, xgboost_args):
    spark = (SparkSession
        .builder
        .appName(args.mainClass)
        .getOrCreate())

    def prepare_data(path):
        reader = spark.read.format(args.format)
        if args.format == 'csv':
            reader.schema(schema).option('header', args.hasHeader)
        return vectorize(reader.load(path), label)

    classifier = (XGBoostClassifier(**merge_dicts(default_params, xgboost_args))
        .setLabelCol(label)
        .setFeaturesCol('features'))
    evaluator = (MulticlassClassificationEvaluator()
        .setLabelCol(label))
    param_grid = (ParamGridBuilder()
        .addGrid(classifier.maxDepth, [5, 10])
        .addGrid(classifier.numRound, [100, 200])
        .build())
    cross_validator = (CrossValidator()
        .setEstimator(classifier)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(param_grid)
        .setNumFolds(3))

    train_data = prepare_data(args.trainDataPath)
    model = cross_validator.fit(train_data)

    spark.stop()
