# Source code for result analysis

## Model comparisons

Code is provided as a Jupyter notebook, see `model_comparisons.ipynb`.

## Plot metric curves

Code is provided as a Jupyter notebook, see `plot_metric_curves.ipynb`.

*To merge evaluation files produced along the training process into one file:*

1. Set variable `EXP_NAMES` in `merge_eval_files.py` to proper value.
2. Run

```sh
python merge_eval_files.py
```
