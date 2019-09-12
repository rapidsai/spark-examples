/* 
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved. 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.rapids.spark.examples.utility

import com.google.common.base.CaseFormat

import scala.collection.mutable
import scala.util.{Try, Success}

private case class XGBoostArg(
  required: Boolean = false,
  isValid: String => Boolean = _ => true,
  message: String = "")

object XGBoostArgs {
  private val modes = Seq("all", "train", "transform")
  private val dataFormats = Seq("csv", "parquet", "orc")
  private val stringToBool = Map(
    "true"  -> true,
    "false" -> false,
    "1" -> true,
    "0" -> false
  )
  private val supportedArgs = Map(
    "format"  -> XGBoostArg(required = true,
      isValid = value => dataFormats.contains(value),
      message = s"Expect one of [${dataFormats.mkString(", ")}]"),
    "mode"    -> XGBoostArg(
      isValid = value => modes.contains(value),
      message = s"Expect one of [${modes.mkString(", ")}]"),
    "modelPath"  -> XGBoostArg(),
    "overwrite"  -> XGBoostArg(
      isValid = value => stringToBool.contains(value),
      message = "Expect 'true' or '1' for true, 'false' or '0' for false."),
    "hasHeader"  -> XGBoostArg(
      isValid = value => stringToBool.contains(value),
      message = "Expect 'true' or '1' for true, 'false' or '0' for false."),
    "evalDataPath"  -> XGBoostArg(),
    "trainDataPath" -> XGBoostArg(),
    "trainEvalDataPath"  -> XGBoostArg(),
    "numRows" -> XGBoostArg(
      isValid = value => Try(value.toInt).isSuccess,
      message = "Require an Int."),
    "showFeatures"  -> XGBoostArg(
      isValid = value => stringToBool.contains(value),
      message = "Expect 'true' or '1' for true, 'false' or '0' for false."),
    "asFloats" -> XGBoostArg(
      isValid = value => stringToBool.contains(value),
      message = "Expect 'true' or '1' for true, 'false' or '0' for false."),
    "maxRowsPerChunk" -> XGBoostArg(
      isValid = value => Try(value.toInt).isSuccess,
      message = "Require an Int.")
  )

  private def help: Unit = {
    println("\n\nSupported arguments:")
    println("    -format=<csv/parquet/orc>: String\n" +
      "        Required. The format of the data, now only supports 'csv', 'parquet' and 'orc'.\n")
    println("    -mode=<all/train/transform>: String\n" +
      "        To control the behavior of apps. Default is 'all'. \n" +
      "        * all: Do training and transformation.\n" +
      "        * train: Do training only, will save model to 'modelPath' if specified.\n" +
      "        * transform: Transformation only, 'modelPath' is required to provide the model.\n")
    println("    -modelPath=path: String\n" +
      "        Specify where to save model after training, or where to load model for transforming only. \n")
    println("    -overwrite=value: Boolean\n" +
      "        Whether to overwrite the current model data under 'modelPath'. Default is false\n")
    println("    -trainDataPath=path: String\n" +
      "        Specify the path of training data(File or Directory). Required when mode is not 'transform'.\n")
    println("    -evalDataPath=path: String\n" +
      "        Specify the path of data for transformation/prediction(File or Directory). Required when mode is not 'train'.\n")
    println("    -trainEvalDataPath=path: String\n" +
      "        Specify the path of data for training with evaluation(File or Directory). Now only one dataset is supported.\n")
    println("    -hasHeader=value: Boolean\n" +
      "        Whether the csv file has header. Default is true.\n")
    println("    -numRows=value: Int\n" +
      "        Number of the rows to show after transformation. Default is 5.\n")
    println("    -showFeatures=value: Boolean\n" +
      "        Whether to include the features columns when showing results of transformation. Default is true.\n")
    println("    -asFloats=value: Boolean\n" +
      "        Whether to cast numerical schema to float schema. Default is true.\n")
    println("    -maxRowsPerChunk=value: Int\n" +
      "        Lines of row to be read per chunk. Default is Integer.MAX_VALUE.\n")
    println("For XGBoost arguments:")
    println("    Now we pass all XGBoost parameters transparently to XGBoost, no longer to verify them.")
    println("    Both of the formats are supported, such as 'numWorkers'. You can pass as either one below:")
    println("    -numWorkers=10  or  -num_workers=10 ")
    println()
  }

  def parse(args: Array[String]): XGBoostArgs = {
    val appArgsMap = mutable.HashMap.empty[String, String]
    val xgbArgsMap = mutable.HashMap.empty[String, String]
    try {
      args.filter(_.nonEmpty).foreach {
        argString =>
          require(argString.startsWith("-") && argString.contains('='),
            s"Invalid argument: $argString, expect '-name=value'")

          val parts = argString.stripPrefix("-").split('=').filter(_.nonEmpty)
          require(parts.length == 2,
            s"Invalid argument: $argString, expect '-name=value'")

          if (supportedArgs.contains(parts(0))) {
            // App arguments
            require(supportedArgs(parts(0)).isValid(parts(1)),
              s"Invalid value to '${parts(0)}'. ${supportedArgs(parts(0)).message}")
            appArgsMap += parts(0) -> parts(1)
          } else {
            // Supposed to be XGBooost parameters
            xgbArgsMap += parts(0) -> parts(1)
          }
      }
      supportedArgs.filter(_._2.required).foreach {
        case (name, _) => require(appArgsMap.contains(name), s"Missing argument: $name.")
      }
      new XGBoostArgs(appArgsMap.toMap, xgbArgsMap.toMap)
    } catch {
      case e: Exception =>
        help
        throw e
    }
  }
}

class XGBoostArgs private[utility] (
    val appArgsMap: Map[String, String],
    val xgbArgsMap: Map[String, String]) {

  // format is required, so on need to check its existence
  def format: String = appArgsMap("format")

  // mode is optional with default value 'all'
  private def mode: String = appArgsMap.getOrElse("mode", "all")

  def isToTrain: Boolean = mode != "transform"
  def isToTransform: Boolean = mode != "train"

  private[utility] def verifyArgsRelation: Unit = {
    if(isToTrain) {
      require(appArgsMap.contains("trainDataPath"), s"'trainDataPath' is required for mode: $mode")
      if (mode == "train" && !appArgsMap.contains("modelPath")) {
        println("==> You may want to specify the 'modelPath' to save the model when 'train only' mode.")
      }
    }
    if(isToTransform) {
      require(appArgsMap.contains("evalDataPath"), s"'evalDataPath' is required for mode: $mode")
      if (mode == "transform") {
        require(appArgsMap.contains("modelPath"), s"'modelPath' is required for mode: $mode")
      }
    }
  }
  verifyArgsRelation

  // trainDataPath and evalDataPath are checked by 'validateArgsRelation'
  // The order is (train, train-eval, eval), only return the necessary data paths based on current mode
  def dataPaths: Seq[Option[String]] = Seq(
    if (isToTrain) appArgsMap.get("trainDataPath") else None,
    appArgsMap.get("trainEvalDataPath"),
    if (isToTransform) appArgsMap.get("evalDataPath") else None)

  def modelPath: Option[String] = appArgsMap.get("modelPath")

  def isOverwrite: Boolean = appArgsMap.get("overwrite")
    .exists(value => XGBoostArgs.stringToBool(value.toString))

  def hasHeader: Boolean = appArgsMap.get("hasHeader")
    .forall(value => XGBoostArgs.stringToBool(value.toString))

  def numRows: Int = appArgsMap.get("numRows").map(_.toInt).getOrElse(5)

  def isShowFeatures: Boolean = appArgsMap.get("showFeatures")
    .forall(value => XGBoostArgs.stringToBool(value.toString))

  def asFloats: Boolean = appArgsMap.get("asFloats")
    .forall(value => XGBoostArgs.stringToBool(value.toString))

  def maxRowsPerChunk: Int = appArgsMap.get("maxRowsPerChunk")
    .map(_.toInt).getOrElse(Integer.MAX_VALUE)

  def xgboostParams(
      otherParams: Map[String, Any] = Map.empty): Map[String, Any] = {
    otherParams ++ xgbArgsMap.map{
        case (name, value) if !name.contains('_') =>
          (CaseFormat.LOWER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, name), value)
        case (name, value) => (name, value)
    }
  }
}


