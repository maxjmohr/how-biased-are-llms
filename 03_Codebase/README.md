# The Codebase
Welcome to the codebase for my master's thesis.

## Dependencies

To run any scripts or contribute to the project, simply install the dependencies via pip or anaconda:

```bash
fynd -m venv .venv
source .venv/bin/activate
fynd -m pip install -r requirements.txt
```

```bash
conda env create -n tue_mthesis -f environment_nobuilds.yml
conda activate tue_mthesis
```

## Pipeline Overview
![pipeline](experiment_cycle.png)