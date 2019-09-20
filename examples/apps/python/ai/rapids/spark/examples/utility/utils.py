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
from pyspark.ml.evaluation import *
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from time import time

def merge_dicts(dict_x, dict_y):
    result = dict_x.copy()
    result.update(dict_y)
    return result

def show_sample(args, data_frame, label):
    data_frame = data_frame if args.showFeatures else data_frame.select(label, 'prediction')
    data_frame.show(args.numRows)

def vectorize(data_frame, label):
    features = [ x.name for x in data_frame.schema if x.name != label ]
    to_floats = [ col(x.name).cast(FloatType()) for x in data_frame.schema ]
    return (VectorAssembler()
        .setInputCols(features)
        .setOutputCol('features')
        .transform(data_frame.select(to_floats))
        .select(col('features'), col(label)))

def with_benchmark(phrase, action):
    start = time()
    result = action()
    end = time()
    print('-' * 100)
    print('{} takes {} seconds'.format(phrase, round(end - start, 2)))
    return result

def check_classification_accuracy(data_frame, label):
    accuracy = (MulticlassClassificationEvaluator()
        .setLabelCol(label)
        .evaluate(data_frame))
    print('-' * 100)
    print('Accuracy is ' + str(accuracy))

def check_regression_accuracy(data_frame, label):
    accuracy = (RegressionEvaluator()
        .setLabelCol(label)
        .evaluate(data_frame))
    print('-' * 100)
    print('RMSE is ' + str(accuracy))
