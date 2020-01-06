This directory contains the code for various latent variable models to summarize the patient state at each point of their trajectory and then cluster the change in state over time. The code in this repository is from a class project with Suchan Vivatsethachai and Liyang (Sophie) Sun.

In this part, we use the answers from the individual questions in parts  II and III of the MDS-UPDRS assessment. If multiple assessments are taken at a visit at different treatment states, we take the maximum answer for each question. We implemented 3 types of models:
- Linear factor analysis
- Factor analysis with the decoder as an ordinal regression
- Variational autoencoder based models. More details and code for this part can be found in the `variational_autoencoder_models` directory.
The first two models are implemented in the `LatentModels` directory. For the latter two, we implement a longitudinal loss function that penalizes the model if the latent state does not increase monotonically with time for each patient.

Next, we project the latent state forwards using a linear regression and predict the observed features using the model. Lastly, we also cluster the patients based on the linear regression model of their trajectories. These analyses are implemented in the `PostLatentModels` directory.

To evaluate these models, we use the following 3 metrics:
- Mean squared error in output at the current timestep
- Mean squared error in output at future timesteps if we project the latent state forwards using a linear regression
- Concordance index to measure whether the latent state increases monotonically across time for each patient
These evaluators are implemented in the `Evaluation` directory.
