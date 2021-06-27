# RobustML

This repository contains the implementation source code of the following paper:

Analyzing and Improving the Robustness of Tabular Classifiers using Counterfactual Explanations

# Setup
1- Clone the repository using HTTP/SSH:
```
git clone https://github.com/peymanrasouli/RobustML
```
2- Install the following package containing GCC/g++ compilers and libraries:
```
sudo apt-get install build-essential
```
3- Create a conda virtual environment:
```
conda create -n RobustML python=3.8
```
4- Activate the conda environment: 
```
conda activate RobustML
```
5- Standing in RobustML directory, install the requirements:
```
pip install -r requirements.txt
```

# Reproducing the robustness analysis results
1- To reproduce the results of the robustness analysis of black-box models run:
```
python robustness_analysis_blackbox.py
```
2- To reproduce the results of the perturbation efficacy benchmark:
```
python perturbation_efficacy_benchmark.py
```

# Reproducing the robustness improvement results
1- To reproduce the results of the robustness improvement of black-box models run:
```
python robustness_improvement_blackbox.py
```
2- To reproduce the results of the robustness improvement of neural network model run:
```
python robustness_improvement_nn.py
```
3- To reproduce the success rate results of baseline methods after robustness improvement run:
```
python robustness_improvement_success_rate.py
```
