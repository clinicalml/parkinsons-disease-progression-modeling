# Clinical trial sample size calculations

We compare the sample sizes that would be needed to detect a significant effect in clinical trials using standard outcomes versus the outcomes we defined. We also compare the sample sizes if we assume the entire cohort is enrolled versus if a predicted subset of patients is enrolled. This analysis is described in Ch. 6.

1. Run `Binary classification (python 3).ipynb` to set up the truncated outcome.
2. Run `Set up test patnos.ipynb` to get a held-out test set that has no missing features in any of the covariate sets.
3. For each of the 4 model types (logistic regression, decision tree, random forest, and survival analysis), run the respective python file with the covariate set and the number of years at which truncationg was set as the parameters.
4. Calculate the sample sizes using the prediction with `Calculate sample sizes for best models test set fixed.ipynb`
