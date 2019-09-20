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

from pyspark.sql.types import *

label = 'fare_amount'

schema = StructType([
    StructField('vendor_id', FloatType()),
    StructField('passenger_count', FloatType()),
    StructField('trip_distance', FloatType()),
    StructField('pickup_longitude', FloatType()),
    StructField('pickup_latitude', FloatType()),
    StructField('rate_code', FloatType()),
    StructField('store_and_fwd', FloatType()),
    StructField('dropoff_longitude', FloatType()),
    StructField('dropoff_latitude', FloatType()),
    StructField(label, FloatType()),
    StructField('hour', FloatType()),
    StructField('year', IntegerType()),
    StructField('month', IntegerType()),
    StructField('day', FloatType()),
    StructField('day_of_week', FloatType()),
    StructField('is_weekend', FloatType()),
])

features = [ x.name for x in schema if x.name != label ]

default_params = {
    'eta': 0.05,
    'maxDepth': 8,
    'subsample': 0.8,
    'gamma': 1.0,
    'numRound': 100,
    'numWorkers': 1,
}
