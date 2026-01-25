# rf_pipeline
Pipeline for Random-Forest models used in the scientific abstract of the European Association for the Study of the Liver Conference 2026. <br><br>
**Scientific Project Summary**:<br>
Portal hypertension is a severe consequence of liver cirrhosis and the main contributor to adverse effects in patients. Non-invasive tests (NIT) are important for patient-stratification and outcome prediction. We developed novel Random Forest-based NITs, the 8P-R and 8P-C models, using blood-based laboratory parameters in a large cohort of patients with advanced chronic liver disease. This repository contains the pipeline used in the scientific abstract titled _**"Machine learning-based 8P-model achieves high accuracy in predicting portal hypertension severity"**_. <br><br>

**Pipeline Summary**:<br>
The pipeline integrates the sklearn implementations of Random Forest Classifiers and Regressors into one function. It further provides:<br>
- Functionality for **imputation** of missing data using sklearns IterativeImputer. It additionally handles mixed data-types (categorical vs. metric). <br>
- Parsing of **model configurations**<br>
- Calculation of **performance metrics**: Categorical Task (Accuracy, F1-Score, Sensitivity, Specificity, Positive-Predictive Value), Regression Task (Pearson's Correlation Coefficient)<br>
- **Output generation**

## Installation
To setup rf_pipeline locally, please create a separate conda environment first:

```bash
conda create -n <env_name> python=3.12
conda activate <env_name>
```

Next, clone the repository, switch to the cloned directory and run pip:

```bash
git clone https://github.com/tsorz/rf_pipeline.git
cd ./rf_pipeline

pip install . # include -e flag to enable editable mode
```

A small test is provided within the rf_tools.py script. Run the script directly to start the test. 
```bash
python rf_tools.py # within the cloned repository
```


