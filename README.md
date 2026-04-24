# kmc — Koopman Model Control

Python library for data-driven control using Koopman operator theory.

## Models
- `DMDc` — Dynamic Mode Decomposition with control
- `EDMDc` — Extended Dynamic Mode Decomposition with control
- `LitKAEc` — Deep Koopman autoencoder with control (PyTorch Lightning)

## Structure
```
kmc/
├── src/kmc/
│   ├── model/       # DMDc, EDMDc, LitKAEc
│   ├── utils/       # Observable functions, model wrappers, MLflow helpers
│   └── base.py      # BaseObservable
└── sysid/           # AUV system identification pipeline (see sysid/README.md)
```

## Installation

```bash
pip install -e .
```

Requires Python 3.10+.
