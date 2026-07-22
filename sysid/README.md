# AUV System Identification Pipeline

Training and validation pipeline for data-driven system identification of an autonomous underwater vehicle (AUV) using Koopman operator methods from the `kmc` library.

## Models
- DMDc — Dynamic Mode Decomposition with control
- EDMDc — Extended Dynamic Mode Decomposition with control
- Deep Koopman with control

## Structure
```
sysid/
├── config/              # Experiment configs (model, data, features)
│   └── kaec/            # Koopman-based model configs
├── notebook/            # ETL prototype notebook
├── container/           # Container definitions
│   ├── Dockerfile
│   └── singularity.def
├── script/              # Entry-point scripts
│   ├── setup.py         # Create MLflow run and log config
│   ├── process.py       # Fetch and preprocess data
│   ├── train.py         # Train model
│   ├── validate.py      # Evaluate and log results
│   └── utils/           # Shared helpers (data loading, metrics)
└── run_local.sh         # Run full pipeline locally
```

## Architecture

```mermaid
flowchart LR
    subgraph local["🖥️  Local / Container"]
        direction TB
        kmc("⚙️ kmc library\nDMDc · EDMDc · LitKAEc")

        subgraph pipeline["  sysid pipeline  "]
            direction TB
            A("📋 setup.py\ncreate MLflow run · log config")
            B("🔄 process.py\nfetch · preprocess · upload")
            C("🏋️ train.py\ntrain model · log model")
            D("📊 validate.py\npredict · log results")
            A --> B --> C --> D
        end

        kmc --> pipeline
    end

    subgraph nas["🗄️  AMARR — Synology NAS"]
        direction TB
        subgraph mlflow["📈  MLflow Tracking Server"]
            direction TB
            runs("Runs / Experiments\nparams · metrics")
            registry("Model Registry")
            art("Artifacts\nplots · scores")
        end

        subgraph s3["🪣  MinIO / S3"]
            direction TB
            raw("raw/\nAUV CSV files")
            clean("cleaned/\ntrain · val · test")
        end
    end

    A -- "log config + run_id" --> runs
    B -- "read" --> raw
    B -- "upload" --> clean
    B -- "log dataset lineage" --> runs
    C -- "read" --> clean
    C -- "log params/metrics" --> runs
    C -- "register model" --> registry
    D -- "load model" --> registry
    D -- "read test data" --> clean
    D -- "log figures/scores" --> art

    style kmc fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style A fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    style B fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    style C fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    style D fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    style runs fill:#fef9c3,stroke:#ca8a04,color:#451a03
    style registry fill:#fef9c3,stroke:#ca8a04,color:#451a03
    style art fill:#fef9c3,stroke:#ca8a04,color:#451a03
    style raw fill:#dcfce7,stroke:#16a34a,color:#052e16
    style clean fill:#dcfce7,stroke:#16a34a,color:#052e16
```

## Prerequisites
- Python 3.10+
- `kmc` package installed (`pip install -e .` from repo root)
- `.env` file with MLflow and S3 credentials (see below)

## Getting Started

1. Clone the repo and install the package:
    ```bash
    git clone <repo-url>
    cd kmc
    pip install -e .
    ```

2. Create a `.env` file inside `sysid/`:
    ```
    MLFLOW_TRACKING_URI=...
    S3_ENDPOINT_URL=...
    S3_ACCESS_KEY_ID=...
    S3_SECRET_ACCESS_KEY=...
    ```

3. Run the full pipeline from `sysid/`:
    ```bash
    cd sysid
    bash run_local.sh
    ```

## Docker

Build and run with GPU support:
```bash
cd kmc
docker build -f sysid/container/Dockerfile -t kmc-sysid .
docker run --gpus all -it --rm -v $(pwd)/sysid:/workspace kmc-sysid
bash run_local.sh
```
