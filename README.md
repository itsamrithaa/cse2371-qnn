# Neural Networks Approaches to Molecular Solubility Predictions


Unified solubility modeling pipeline comparing a CNN, a pure QNN, and a hybrid classical+QNN model. Everything (data cleaning, training, evaluation on four chemistry-driven series, and metric comparison) now lives in a single notebook:

1. Create a virtual environment (recommended) and install deps inside it:
   - macOS/Linux: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
   - Windows: `python -m venv .venv && .venv\\Scripts\\activate && pip install -r requirements.txt`
   - Use Python 3.10+.
2. Launch Jupyter from the activated env: `jupyter notebook` (or use VS Code notebooks) and pick the `.venv` kernel if prompted.
3. Run `solubility/notebooks/solubility_pipeline.ipynb` top to bottom. It will:
   - Clean/canonicalize AqSolDB and create train/val/test splits.
   - Train CNN, QNN, and Hybrid; save weights to `solubility/artifacts/` and metrics/predictions to `solubility/results/`.
   - Write `model_metrics.csv` and comparison plots.
4. For quick inference/visual checks on predefined series using saved weights, run `solubility/notebooks/series_test.ipynb` after the pipeline.
5. Optional: the CNN+Descriptors variant (pooled CNN features + RDKit descriptors) is defined in `cnn_utils.py` with a training cell near the end of the pipeline notebook; run it if you need that model’s metrics.

Legacy exploratory notebooks are tucked under `solubility/notebooks/archive/` to keep the workspace tidy while preserving past work.

## References
`[1] Abbas AH (2025). TunnElQNN: A hybrid quantum-classical neural network for efficient learning. arXiv preprint arXiv:2505.00933. https://doi.org/10.48550/arXiv.2505.00933` 

`[2] Delaney JS (2004). ESOL: estimating aqueous solubility directly from molecular structure. J Chem Inf Comput Sci 44(3):1000–1005. https://pubs.acs.org/doi/10.1021/ci034243x`

`[3] de Lima GG, Farias TS, Ricardo AC, Boas CJV (2025). Assessing the advantages and limitations of quantum neural networks in regression tasks. Federal University of Sao Carlos. https://doi.org/10.48550/arXiv.2509.00854`

`[4] Ghanavati MA, Ahmadi S, Rohani S (2024). A machine learning approach for the prediction of aqueous solubility of pharmaceuticals: a comparative model and dataset analysis. Digital Discovery 3:2085–2104. https://doi.org/10.1039/D4DD00065J`

`[5] Liu J, Lei X, Ji C, Pan Y (2023). Fragment-pair based drug molecule solubility prediction through attention mechanism. Front Pharmacol. https://doi.org/10.3389/fphar.2023.1255181`

`[6] O’Boyle NM (2012). Towards a universal SMILES representation: a standard method to generate canonical SMILES based on the InChI. Journal of Cheminformatics. https://doi.org/10.1186/1758-2946-4-22`

`[7] Potempa R, Porebski S (2025). Comparing concepts of quantum and classical neural network models for image classification tasks. Silesian University of Technology. https://doi.org/10.1007/978-3-030-81523-3_6`

`[8] Sorkun, M.C., Khetan, A. & Er, S. AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. Sci Data 6, 143 (2019). https://doi.org/10.1038/s41597-019-0151-1`
