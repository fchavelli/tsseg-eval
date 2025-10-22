# Solving dependency issues:

## RuntimeError: cannot load _umath_tests module.
```bash
export NUMPY_EXPERIMENTAL_DTYPE_API=1
```
Then redo the experiment (e.g.):
```bash
python experiments.py multivariate --a hsphsmm --d MoCap
```

## ImportError: cannot import name 'logsumexp' from 'scipy.misc'
```bash
nano /home/username/miniconda3/envs/env/lib/python3.9/site-packages/pybasicbayes/models/mixture.py
```
Change the following line:
```bash
from scipy.misc import logsumexp
```
to
```bash
from scipy.special import logsumexp
```

Same for `negativebinomial.py`