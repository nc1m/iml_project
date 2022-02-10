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
    ├── resources
    │   ├── lecture_slides    
    │   └── paper                         
    ├── src                     
    │   ├── integrated_gradients.py
    │   └── paper  
    │
    └── requirements.txt          # Dependencies
