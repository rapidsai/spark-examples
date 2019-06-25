package ai.rapids.spark.examples.mortgage

import org.apache.spark.sql.types.{FloatType, IntegerType, StructField, StructType}

private[mortgage] trait Mortgage {
  val labelColName = "delinquency_12"

  val schema = StructType(List(
    StructField("orig_channel", FloatType),
    StructField("first_home_buyer", FloatType),
    StructField("loan_purpose", FloatType),
    StructField("property_type", FloatType),
    StructField("occupancy_status", FloatType),
    StructField("property_state", FloatType),
    StructField("product_type", FloatType),
    StructField("relocation_mortgage_indicator", FloatType),
    StructField("seller_name", FloatType),
    StructField("mod_flag", FloatType),
    StructField("orig_interest_rate", FloatType),
    StructField("orig_upb", IntegerType),
    StructField("orig_loan_term", IntegerType),
    StructField("orig_ltv", FloatType),
    StructField("orig_cltv", FloatType),
    StructField("num_borrowers", FloatType),
    StructField("dti", FloatType),
    StructField("borrower_credit_score", FloatType),
    StructField("num_units", IntegerType),
    StructField("zip", IntegerType),
    StructField("mortgage_insurance_percent", FloatType),
    StructField("current_loan_delinquency_status", IntegerType),
    StructField("current_actual_upb", FloatType),
    StructField("interest_rate", FloatType),
    StructField("loan_age", FloatType),
    StructField("msa", FloatType),
    StructField("non_interest_bearing_upb", FloatType),
    StructField(labelColName, IntegerType)))

  val commParamMap = Map(
    "eta" -> 0.1,
    "gamma" -> 0.1,
    "missing" -> 0.0,
    "max_depth" -> 10,
    "max_leaves" -> 256,
    "grow_policy" -> "depthwise",
    "min_child_weight" -> 30,
    "lambda" -> 1,
    "scale_pos_weight" -> 2,
    "subsample" -> 1,
    "nthread" -> 1,
    "num_round" -> 100)
}