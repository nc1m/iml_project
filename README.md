<!--# iml_project -->
<h2 align="left">
Integrated Gradients: Baselines for Text Classification
</h2>

Many explanation methods, including the popular `Integrated Gradients` method, require choosing a baseline hyperparameter. What does this parameter mean and how does it impact the resulting explanations? What implicit assumptions are made by selecting different baselines?
In this project, the goal is to investigate these questions using text classification models as a case study.

### Dependencies

Dependencies can be installed using `requirements.txt`

### Installation instruction
To install the requirements, follow these steps:
```bash
# Install requirements
pip install -r requirements.txt

```

### Usage
The default branch for latest and stable changes is `main`.
```bash
# Run file
python example.py

```

### What does each file do?

    .
    ├── scripts
    │   └── run.py                # The main runner script to launch jobs.
    ├── src                     
    │   ├── agent.py              # Implements the Agent API for action selection 
    │   ├── algos.py              # Distributional RL loss
    │   ├── models.py             # Network architecture and forward passes.
    │   ├── rlpyt_atari_env.py    # Slightly modified Atari env from rlpyt
    │   ├── rlpyt_utils.py        # Utility methods that we use to extend rlpyt's functionality
    │   └── utils.py              # Command line arguments and helper functions 
    │
    └── requirements.txt          # Dependencies
