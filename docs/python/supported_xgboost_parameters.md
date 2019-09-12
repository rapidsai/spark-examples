Supported XGBoost Parameters
============================

This is a description of all the parameters available when you are running examples in this repo:

1. All [xgboost parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) are supported.
   * Please use the `camelCase`, e.g., `--treeMethod=gpu_hist`.
   * `lambda` is replaced with `lambda_`, because `lambda` is a keyword in Python.
2. `--format=[csv|parquet|orc]`: The format of the data for training/transforming, now supports 'csv', 'parquet' and 'orc'. *Required*.
3. `--mode=[all|train|transform]`. To control the behavior of the sample app, default is 'all' if not specified.
   * all: Do both training and transforming, will save model to 'modelPath' if specified
   * train: Do training only, will save model to 'modelPath' if specified.
   * transform: Do transforming only, 'modelPath' is required to locate the model data to be loaded.
4. `--trainDataPath=[path]`: Path to your training data file(s), required when mode is NOT 'transform'.
5. `--trainEvalDataPath=[path]`: Path to your data file(s) for training with evaluation. Optional.
6. `--evalDataPath=[path]`: Path to your test(evaluation) data file(s), required when mode is NOT 'train'.
7. `--modelPath=[path]`: Path to save model after training, or where to load model for transforming only. Required only when mode is 'transform'.
8. `--overwrite=[true|false]`: Whether to overwrite the current model data under 'modelPath'. Default is false. You may need to set to true to avoid IOException when saving the model to a path already exists.
9. `--hasHeader=[true|false]`: Indicate if your csv file has header.
10. `--asFloats=[true|false]`: Whether to cast numerical schema to float schema. Default is true.
11. `--maxRowsPerChunk=[value]`: Max lines of row to be read per chunk. Default is 2147483647.
