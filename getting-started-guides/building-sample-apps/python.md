# Build XGBoost Python Examples

##### Build Process

Follow these steps to package the Python zip file:

```
git clone https://github.com/rapidsai/spark-examples.git
cd spark-examples/examples/apps/python
zip -r samples.zip ai
```

##### Files Required by PySpark

Two files are required by PySpark:

+ *samples.zip* : the package including all example code
+ *main.py*: entrypoint for PySpark, you may just copy it from folder *spark-examples/examples/apps/python*
