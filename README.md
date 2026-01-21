# rf_pipeline
Pipeline for Random-Forest models used in the scientific abstract of European Association for the Study of the Liver. <br><br>
**Project Summary**:<br>
Portal hypertension is a severe consequence of liver cirrhosis and the main contributor to adverse effects in patients. Non-invasive tests (NIT) are important for patient-stratification and outcome prediction. We developed a novel NIT, the 8P-R and 8P-C model, using blood-based laboratory parameters in a large cohort of patients with advanced chronic liver disease. This repository contains the pipeline used in the scientific abstract titled _**"Machine learning-based 8P-model achieves high accuracy in predicting portal hypertension severity"**_. <br><br>

**Pipeline Summary**:<br>
The pipeline integrates the sklearn implementations of Random Forest Classifiers and Regressors into a one function. It further provides:<br>
&nbsp;&nbsp;- Functionality for **imputation** of missing data using sklearns IterativeImputer. It additionally handles mixed data-types (categorical vs. metric). 
&nbsp;&nbsp;- Parsing of **model configurations**<br>
&nbsp;&nbsp;- Calculation of **performance metric**: Categorical Task (Accuracy, F1-Score, Sensitivity, Specificity, Positive-Predictive Value), Regression Task (Pearson's Correlation Coefficient)<br>
&nbsp;&nbsp;- **Output generation**

