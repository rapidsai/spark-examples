import sbtassembly.AssemblyPlugin.autoImport._

val sparkVersion = "2.4.3"
val xgBoostVersion = "0.90"
val scalatestVersion = "3.0.5"

lazy val commonSettings = Seq(
  resolvers += Resolver.mavenLocal,
  organization := "ai.rapids.sparkexamples",
  name := "spark-examples",
  description := "RAPIDS Spark Examples",
  scalaVersion := "2.11.12",
  javacOptions ++= Seq("-source", "1.8", "-target", "1.8", "-Xlint:unchecked"),
  javacOptions in (Compile, doc) := Seq("-source", "1.8"),
  assemblyMergeStrategy in assembly := {
    case s if s.endsWith(".class") => MergeStrategy.last
    case s if s.endsWith(".xsd") => MergeStrategy.last
    case s if s.endsWith(".dtd") => MergeStrategy.last
    case s if s.endsWith("BUILD") => MergeStrategy.last
    case s if s.endsWith("pom.properties") => MergeStrategy.last
    case s if s.endsWith("pom.xml") => MergeStrategy.last
    case s if s.endsWith("cmdline.arg.info.txt.1") => MergeStrategy.last
    case s if s.endsWith("version.properties") => MergeStrategy.last
    case s if s.endsWith("log4j.properties") => MergeStrategy.last
    case s if s.endsWith("properties") => MergeStrategy.last
    case s if s.endsWith("parquet.thrift") => MergeStrategy.first
    case s if s.endsWith("plugin.xml") => MergeStrategy.discard
    case s if s.endsWith("io.netty.versions.properties") => MergeStrategy.discard
    case x =>
      val oldStrategy = (assemblyMergeStrategy in assembly).value
      oldStrategy(x)
  },
)

lazy val root: Project = project.in(file(".")).settings(commonSettings).aggregate(mortgage)

lazy val mortgage: Project = project
  .in(file("mortgage"))
  .settings(commonSettings)
  .settings(
    name := "mortgage",
    moduleName := "mortgage",
    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % scalatestVersion % "test",
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-catalyst" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-tags" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "org.apache.spark" %% "spark-unsafe" % sparkVersion,
      "ml.dmlc" % "xgboost4j" % xgBoostVersion,
      "ml.dmlc" % "xgboost4j-spark" % xgBoostVersion
    )
  )
