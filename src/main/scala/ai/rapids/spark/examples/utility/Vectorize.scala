package ai.rapids.spark.examples.utility

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.FloatType

object Vectorize {
  def apply(df: DataFrame, labelName: String, changeLabelName: Boolean = true): DataFrame = {
    val features = df.schema.collect{case f if f.name != labelName => f.name}
    val toFloat = df.schema.map(f => col(f.name).cast(FloatType))
    val labelCol = if (changeLabelName) col(labelName).alias("label") else col(labelName)
    new VectorAssembler()
      .setInputCols(features.toArray)
      .setOutputCol("features")
      .transform(df.select(toFloat:_*))
      .select(col("features"), labelCol)
  }

  def apply(df: DataFrame, featureNames: Seq[String], labelName: String): DataFrame = {
    val toFloat = df.schema.map(f => col(f.name).cast(FloatType))
    new VectorAssembler()
      .setInputCols(featureNames.toArray)
      .setOutputCol("features")
      .transform(df.select(toFloat:_*))
      .select(col("features"), col(labelName))
  }
}
