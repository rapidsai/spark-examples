#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
from ai.rapids.spark.examples.agaricus.consts import *
from ai.rapids.spark.examples.utility.args import parse_arguments
from ai.rapids.spark.examples.utility.utils import *
from ml.dmlc.xgboost4j.scala.spark import *
from ml.dmlc.xgboost4j.scala.spark.rapids import GpuDataReader
from pyspark.sql import SparkSession

def main(args, xgboost_args):
    spark = (SparkSession
        .builder
        .appName(args.mainClass)
        .getOrCreate())

    def prepare_data(path):
        reader = (GpuDataReader(spark)
            .format(args.format)
            .option('asFloats', args.asFloats)
            .option('maxRowsPerChunk', args.maxRowsPerChunk))
        if args.format == 'csv':
            reader.schema(schema).option('header', args.hasHeader)
        return reader.load(path)

    if args.mode in [ 'all', 'train' ]:
        classifier = (XGBoostClassifier(**merge_dicts(default_params, xgboost_args))
            .setLabelCol(label)
            .setFeaturesCols(features))

        if args.trainEvalDataPath:
            train_eval_data = prepare_data(args.trainEvalDataPath)
            classifier.setEvalSets({ 'test': train_eval_data })

        train_data = prepare_data(args.trainDataPath)
        model = with_benchmark('Training', lambda: classifier.fit(train_data))

        if args.modelPath:
            writer = model.write().overwrite() if args.overwrite else model
            writer.save(args.modelPath)
    else:
        model = XGBoostClassificationModel().load(args.modelPath)

    if args.mode in [ 'all', 'transform' ]:
        eval_data = prepare_data(args.evalDataPath)

        def transform():
            result = model.transform(eval_data).cache()
            result.foreachPartition(lambda _: None)
            return result

        result = with_benchmark('Transformation', transform)
        show_sample(args, result, label)
        with_benchmark('Evaluation', lambda: check_classification_accuracy(result, label))

    spark.stop()
