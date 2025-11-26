# cse2371-qnn

Unified solubility modeling pipeline comparing a CNN, a pure QNN, and a hybrid classical+QNN model. Everything (data cleaning, training, evaluation on four chemistry-driven series, and metric comparison) now lives in a single notebook:

- `solubility/notebooks/solubility_pipeline.ipynb` – run top to bottom to reproduce splits, train all three models, save weights, and export metrics/predictions to `solubility/results/`.

Legacy exploratory notebooks are tucked under `solubility/notebooks/archive/` to keep the workspace tidy while preserving past work.
