In this directory, we apply the variational autoencoder model created in [1] to the PPMI data. With their model, we can obtain a single or a few latent variables to summarize a patient's state at each timepoint. Their model enforces monotonicity among the latent variables across time so this latent state can represent the patient's disease severity.

[1] Pierson, Emma, et al. "Inferring Multidimensional Rates of Aging from Cross-Sectional Data." Proceedings of machine learning research 89 (2019): 97.
The python files are from their repository: https://github.com/epierson9/multiphenotype_methods
