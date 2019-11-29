# Survival outcome definition and models

To create the survival outcomes:
`python Specs_for_subtotals_outcome.py` data_directory<br>
`python Create_Survival_Outcome.py` outcome_directory data_directory num_years prop_pop

Parameters:<br>
data_directory: path to extracted PPMI data<br>
outcome_directory: path where outcome will be outputted to<br>
num_years and prop_pop: One of the outcome requirements will be that it takes at least num_years (number of years) for prop_pop (proportion of the de novo PD cohort) to cross the defined threshold.

To run the survival models:
1. Create the sets of baseline covariates using `Make final covariate sets nested.ipynb`
2. Select a set of patients who have no missing features for any of the baseline covariate sets as the held-out test set using `Unified test set for all covariate sets.ipynb`
3. Run `Final_survival_model.py` with the name of the covariate set as the only parameter.
