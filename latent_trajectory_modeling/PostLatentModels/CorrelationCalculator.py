import numpy as np, pandas as pd, seaborn as sns, matplotlib, pickle, math
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import sys, os
DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DIR_PATH)
from Util.visualizer import *


'''
Calculates the correlation between latent factors and observed features.
'''

class CorrelationCalculator(object):
    
    def calculate_correlations(self, dataframe, latent_column_names, observed_column_names, \
                               human_readable_observed_column_names, plot_filepath, dict_filepath, model_name, \
                               agg_backend=False, xlabels=None, show_ylabels=True, show_cbar=True):
        '''
        Plots heatmap using above table to plot_filepath.
        Stores a dictionary mapping each latent column name to a list of the 5 most correlated features (positive or negative).
        Each list will consist of tuples: (feat_name, correlation)
        '''
        assert set(latent_column_names).issubset(set(dataframe.columns.values.tolist()))
        assert set(observed_column_names).issubset(set(dataframe.columns.values.tolist()))
        assert len(observed_column_names) == len(human_readable_observed_column_names)
        human_readable_dict = dict()
        for idx in range(len(observed_column_names)):
            human_readable_dict[observed_column_names[idx]] = human_readable_observed_column_names[idx]
        human_readable_dataframe = dataframe.rename(columns=human_readable_dict)
        corr_matrix_heatmap(human_readable_dataframe, latent_column_names, human_readable_observed_column_names, plot_filepath, \
                            agg_backend=agg_backend, xlabels=xlabels, show_ylabels=show_ylabels, show_cbar=show_cbar)
        # Calculate the 5 observed features that most correlate (by absolute value) with each latent factor
        corr = np.empty((len(observed_column_names), len(latent_column_names)))
        for latent_idx in range(len(latent_column_names)):
            for observed_idx in range(len(observed_column_names)):
                corr[observed_idx, latent_idx], _ = pearsonr(dataframe[latent_column_names[latent_idx]].values, \
                                                          dataframe[observed_column_names[observed_idx]].values)
        corr_df = pd.DataFrame(corr, columns=latent_column_names)
        corr_df['Observed features'] = human_readable_observed_column_names
        corr_df = corr_df.set_index('Observed features')
        abs_corr = np.absolute(corr)
        top5_dict = dict()
        for latent_idx in range(len(latent_column_names)):
            sorted_idxs = np.argsort(abs_corr[:,latent_idx])
            top5_list = []
            for i in range(5):
                feat_idx = sorted_idxs[int(-1*-i)]
                top5_list.append((human_readable_observed_column_names[feat_idx], corr[feat_idx,latent_idx]))
            top5_dict[latent_column_names[latent_idx]] = top5_list
        if sys.version_info[0] == 3:
            write_tag = 'wb'
        else:
            write_tag = 'w'
        with open(dict_filepath, write_tag) as f:
            pickle.dump(top5_dict, f)

    def plot_feature_r_corr_matrix(self, model, df, observed_column_names, num_r, agg_backend=False):
        df = df.copy().reset_index(drop=True)
        # num_r = model.k_age
        estimated_Zs = model.get_projections(df, project_onto_mean=True)
        for i in range(num_r):
            preprocessed_ages = model.age_preprocessing_function(df['age_sex___age'])
            estimated_r = estimated_Zs['z%i' % i] / preprocessed_ages
            df = df.join(pd.DataFrame({"r{}".format(i): estimated_r}))
        corr_matrix_heatmap(df, ["r{}".format(i) for i in range(num_r)], observed_column_names, agg_backend=agg_backend)

    def get_obs_to_latent_corr(self, df, latent_column_names, observed_column_names):
        """
        calculate top 5 observed features associated with each latent factor (pos + neg)

        df containing latent columns and observed columns
        latent_column_names - column names of the latent variables
        observed_column_names - column names of the raw features
        """
        print('Latent')
        print(len(latent_column_names))
        print(latent_column_names)
        print('Observed')
        print(len(observed_column_names))
        print(observed_column_names)
        print('Dataframe columns')
        print(len(df.columns.values))
        print(df.columns.values)
        assert set(latent_column_names).issubset(set(df.columns.values.tolist()))
        assert set(observed_column_names).issubset(set(df.columns.values.tolist()))
        corr_matrix = np.empty((len(latent_column_names), len(observed_column_names)))
        output_str = ''
        for latent_idx in range(len(latent_column_names)):
            for obs_idx in range(len(observed_column_names)):
                corr_matrix[latent_idx, obs_idx], _ \
                    = pearsonr(df[latent_column_names[latent_idx]].values, \
                               df[observed_column_names[obs_idx]].values)
            sorted_idxs = np.argsort(np.abs(corr_matrix[latent_idx]))
            output_str += latent_column_names[latent_idx] + ' correlated features: '
            for i in range(10):
                obs_idx = sorted_idxs[int(-1*i)]
                output_str += observed_column_names[obs_idx] + ' ({0:.4f}), '.format(corr_matrix[latent_idx, obs_idx])
            output_str = output_str[:-2] + '\n'
        return output_str







