// Databricks notebook source
class ConfigurableWordCount(override val uid: String) extends Transformer {
  
  final val inputCol= new Param[String](this, "inputCol", "The input column")
  final val outputCol = new Param[String](this, "outputCol", "The output column")

 ; def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  def this() = this(Identifiable.randomUID("configurablewordcount"))

  def copy(extra: ParamMap): ConfigurableWordCount = {
    defaultCopy(extra)
  }
  
  override def transformSchema(schema: StructType): StructType = {
    // Check that the input type is a string
    val idx = schema.fieldIndex($(inputCol))
    val field = schema.fields(idx)
    if (field.dataType != StringType) {
      throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
    }
    // Add the return field
    schema.add(StructField($(outputCol), IntegerType, false))
  }
  
  def transform(df: Dataset[_]): DataFrame = {
    val wordcount = udf { in: String => in.split(" ").size }
    df.select(col("*"), wordcount(df.col($(inputCol))).as($(outputCol)))
  }
}
  

// COMMAND ----------

val dataset = spark.read.table("kdd")
display(dataset)

// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer

val protocol_type_indexer = new StringIndexer()
  .setInputCol("protocol_type")
  .setInputCol("protocol_type_index")

val service_indexer = new StringIndexer()
  .setInputCol("service")
  .setInputCol("service_index")

val flag_indexer = new StringIndexer()
  .setInputCol("flag")
  .setInputCol("flag_index")