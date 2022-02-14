<!--# iml_project -->
<h2 align="left">
Integrated Gradients: Baselines for Text Classification
</h2>

Many explanation methods, including the popular `Integrated Gradients` method, require choosing a baseline hyperparameter. What does this parameter mean and how does it impact the resulting explanations? What implicit assumptions are made by selecting different baselines?
In this project, the goal is to investigate these questions using text classification models as a case study.
To visualize the effects of different baselines we created a graphic inspired by the Distill publication "Visualizing the Impact of Feature Attribution Baselines" by Sturmfels et. al. (see https://distill.pub/2020/attribution-baselines/#figure4_div).

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

To see a list of commandline options run:
```bash

python main.py -h

```

### What does each file do?

    .
    ├── resources
    │   ├── lecture_slides    
    │   └── paper                         
    ├── src                     
    │   ├── baselines.py                    # construct different baselines
    │   └── bert_datasets.py                
    │   └── customized_ig.py                #customized integraded gradients
    │   └── customized_lig.py               #customized layer integraded gradients
    │   └── utils.py                            
    │   └── test_baselines.py               # unittest for baselines
    ├── main.py                             # main
    └── requirements.txt                    # Dependencies
