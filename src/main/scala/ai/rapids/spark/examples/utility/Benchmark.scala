package ai.rapids.spark.examples.utility

object Benchmark {
  def time[R](phase: String)(block: => R): (R, Float) = {
    val t0 = System.currentTimeMillis
    val result = block // call-by-name
    val t1 = System.currentTimeMillis
    println("\n--------------")
    println("==> Benchmark: Elapsed time for [" + phase + "]: " + ((t1 - t0).toFloat / 1000) + "s")
    println("--------------\n")
    (result, (t1 - t0).toFloat / 1000)
  }
}
