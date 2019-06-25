package ai.rapids.spark.examples.taxi

import org.apache.spark.sql.types.{FloatType, IntegerType, StructField, StructType}

private[taxi] trait Taxi {
  val labelColName = "fare_amount"

  lazy val commParamMap = Map(
    "learning_rate" -> 0.05,
    "max_depth" -> 8,
    "subsample" -> 0.8,
    "gamma" -> 1
  )

  lazy val schema =
    StructType(Array(
      StructField("vendor_id", FloatType),
      StructField("passenger_count", FloatType),
      StructField("trip_distance", FloatType),
      StructField("pickup_longitude", FloatType),
      StructField("pickup_latitude", FloatType),
      StructField("rate_code", FloatType),
      StructField("store_and_fwd", FloatType),
      StructField("dropoff_longitude", FloatType),
      StructField("dropoff_latitude", FloatType),
      StructField(labelColName, FloatType),
      StructField("hour", FloatType),
      StructField("year", IntegerType),
      StructField("month", IntegerType),
      StructField("day", FloatType),
      StructField("day_of_week", FloatType),
      StructField("is_weekend", FloatType)
    ))
}
