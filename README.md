# Modeling progression of Parkinson's disease

First, we process data from the Parkinson’s Progression Markers Initiative (PPMI) to convert it into a format that is easier to use for downstream machine learning analyses. Utilizing this data, we design novel data-driven outcomes that capture impairment in motor, cognitive, autonomic, psychiatric, and sleep symptoms and allow for heterogeneity in the patient population. Then, we build survival analysis models to predict these outcomes from baseline. Using our motor and hybrid outcomes can reduce the sample sizes and enrollment time for early PD clinical trials. We can provide further reductions by identifying more severe patients for enrollment via survival analysis and binary classification methods. For summarizing patient state, we seek better representations of disease burden by learning trajectories of disease progression. Lastly, we consider ways to use these patient representations and outcomes for discovering subtypes that capture differing rates of progression.

This work is associated with Christina Ji's MEng thesis Modeling progression of Parkinson's disease. If you have any questions about the thesis or code, please feel free to email cji@mit.edu

Directories:<br>
The `ppmi_extraction` directory contains the scripts for extracting PPMI data into ready-to-use csv formats. The PPMI data is available at https://www.ppmi-info.org/. A description of our data processing can be found in Ch. 2 of the thesis.

We define novel outcomes for Parkinson's disease to capture changes in motor, cognitive, autonomic, psychiatric, and sleep symptoms. We also define a hybrid outcome. These definitions are in Ch. 4 of the thesis. Next, we set up survival models to predict these outcomes from baseline features, as described in Ch. 5. The code for this part is in the `survival_model` directory.

We compare the sample sizes that would be needed to detect a significant effect in clinical trials using standard outcomes versus the outcomes we defined. We also compare the sample sizes if we assume the entire cohort is enrolled versus if a predicted subset of patients is enrolled. This analysis is described in Ch. 6. The code for this part is in the `clinical_trial_sample_sizes` directory.

In the `learning_patient_trajectories` directory, we learn trajectories for subtotals of the MDS-UPDRS II + III motor assessment. Specifically, we look at the 4 treatment settings (untreated, levodopa on phase, levodopa off phase, and MAO-B inhibitor only). Then, for each patient, we fit linear, piecewise linear, and quadratic models to the subtotals. Additional details can be found in Sec. 7.1 of the thesis.

Sec. 7.2 and 8.1 are joint work with Suchan Vivatsethachai and Sophie Sun in a class project. The code for those parts are in another repository.

We take two approaches to subtyping: k-means and non-negative matrix factorization. Our non-negative matrix factorization experiments are built on code from https://github.com/ffaghri1/PD-progression-ML. This part is in the `subtyping` directory. More details can be found in Sec. 8.2 and 8.3 of the thesis.

Dependencies: jupyter, numpy, pandas, matplotlib, sklearn, pickle, lifelines, scipy, seaborn