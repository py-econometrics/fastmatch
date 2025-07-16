# fastmatch


Fast matching estimators for causal inference under unconfoundedness. Interpretable, fast alternative to doubleML under unconfoundedness and overlap.

Supports
- *Estimand*: ATT, ATE
- *Estimator*: vanilla KNN, bias-corrected KNN (the latter lets you use standard errors using Rai/Otsu 2017)
- *Nearest Neighbours Solvers*: scikit, faiss, voyager
  - the latter two are very fast and can be used with GPU support, and can also be exported to a production environment where adding new units and future nearest-neighbours queries are very fast

Uses modern vectorDBs for knn search
- [`faiss`](https://github.com/facebookresearch/faiss)
- [`voyager`](https://github.com/spotify/voyager)


To install, run

```
pip install git+https://github.com/py-econometrics/fastmatch.git
```

Install with gpu support: `pip install fastmatch[gpu]`

