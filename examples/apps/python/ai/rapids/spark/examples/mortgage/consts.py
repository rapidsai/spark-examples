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

label = 'delinquency_12'

schema = StructType([
    StructField('orig_channel', FloatType()),
    StructField('first_home_buyer', FloatType()),
    StructField('loan_purpose', FloatType()),
    StructField('property_type', FloatType()),
    StructField('occupancy_status', FloatType()),
    StructField('property_state', FloatType()),
    StructField('product_type', FloatType()),
    StructField('relocation_mortgage_indicator', FloatType()),
    StructField('seller_name', FloatType()),
    StructField('mod_flag', FloatType()),
    StructField('orig_interest_rate', FloatType()),
    StructField('orig_upb', IntegerType()),
    StructField('orig_loan_term', IntegerType()),
    StructField('orig_ltv', FloatType()),
    StructField('orig_cltv', FloatType()),
    StructField('num_borrowers', FloatType()),
    StructField('dti', FloatType()),
    StructField('borrower_credit_score', FloatType()),
    StructField('num_units', IntegerType()),
    StructField('zip', IntegerType()),
    StructField('mortgage_insurance_percent', FloatType()),
    StructField('current_loan_delinquency_status', IntegerType()),
    StructField('current_actual_upb', FloatType()),
    StructField('interest_rate', FloatType()),
    StructField('loan_age', FloatType()),
    StructField('msa', FloatType()),
    StructField('non_interest_bearing_upb', FloatType()),
    StructField(label, IntegerType()),
])

features = [ x.name for x in schema if x.name != label ]

default_params = {
    'eta': 0.1,
    'gamma': 0.1,
    'missing': 0.0,
    'maxDepth': 10,
    'maxLeaves': 256,
    'growPolicy': 'depthwise',
    'objective': 'binary:logistic',
    'minChildWeight': 30.0,
    'lambda_': 1.0,
    'scalePosWeight': 2.0,
    'subsample': 1.0,
    'nthread': 1,
    'numRound': 100,
    'numWorkers': 1,
}
