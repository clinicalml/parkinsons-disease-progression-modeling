import numpy as np, pandas as pd, seaborn as sns, matplotlib, pickle
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import sys, os
DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DIR_PATH)
from Util.visualizer import *

'''
Calculates the rate of latent factors
'''

class LatentRateCalculator(object):
    
    def get_rate_of_latent(self, dataFrame, latent_column_names, fit_intercept=True):
        '''
        dataFrame has PATNO, EVENT_ID_DUR and latent_column_names as columns
        all patients must have at least 2 timepoints
        latent_column_names is a list of column names corr. latent factors
        Fit a linear regression to each of the latent factors per patient
        Output a dataframe of [PATNO, latentcol_rate, latentcol_bias]
        '''
        col_set = {'PATNO','EVENT_ID_DUR'} | set(latent_column_names)
        assert col_set.issubset(set(dataFrame.columns.values.tolist()))
        assert len(dataFrame.dropna(subset=['PATNO','EVENT_ID_DUR']+latent_column_names)) == len(dataFrame)
        pat_counts = dataFrame.PATNO.value_counts()
        assert min(pat_counts.values) >= 2
        pat_list = dataFrame.PATNO.unique()
        df = pd.DataFrame({'PATNO': pat_list}) 

        for col in latent_column_names:
            rate = []
            bias = []
            for i in pat_list:
                x = dataFrame['EVENT_ID_DUR'][dataFrame['PATNO']==i].values
                y = dataFrame[col][dataFrame['PATNO']==i].values
                linreg = LinearRegression(fit_intercept=fit_intercept,).fit(x.reshape(-1,1), y)
                rate.append(linreg.coef_[0])
                bias.append(linreg.intercept_)
            rate_col = col+'_rate'
            bias_col = col+'_bias'
            rate_col_df = pd.DataFrame({rate_col: rate, bias_col: bias}) 
            df = df.join(rate_col_df) 
        return df
    
    def get_atleast_2timepoints_patnos_dataframe(self, dataFrame):
        '''
        only keep patients with at least 2 datapoints
        '''
        patno_counts = dict(dataFrame.PATNO.value_counts())
        patnos_atleast_2timepoints = set()
        for patno in patno_counts.keys():
            if patno_counts[patno] >= 2:
                patnos_atleast_2timepoints.add(patno)
        return dataFrame.loc[dataFrame['PATNO'].isin(patnos_atleast_2timepoints)]
    
    def plot_latent_factors(self, dataFrame, latent_col, path_to_file, agg_backend=False):
        '''
        dataFrame has PATNO, EVENT_ID_DUR and latent_col as columns
        randomly selects a subset of 20 patients to plot with time on x-axis and latent factor on y-axis
        saves figure to path_to_file
        '''
        assert {'PATNO', 'EVENT_ID_DUR', latent_col}.issubset(set(dataFrame.columns.values.tolist()))
        plot_latent_by_time(dataFrame, 'PATNO', 'EVENT_ID_DUR', latent_col, path_to_file=path_to_file, agg_backend=agg_backend)

    def extrapolate_latent_factors(self, latent_dataframe, future_times_dataframe, latent_column_names):
        '''
        latent_dataframe has PATNO, latentcol_rate, latentcol_bias as columns
        future_times_dataframe: has PATNO, EVENT_ID_DUR as columns
        returns dataframe with first 2 columns matching future_times_dataframe and 3rd column is latent factor at that time
            calculated using linear regression in latent_dataframe
        '''
        assert 'PATNO' in set(latent_dataframe.columns.values.tolist())
        assert set([col + '_rate' for col in latent_column_names]).issubset(set(latent_dataframe.columns.values.tolist()))
        assert set([col + '_bias' for col in latent_column_names]).issubset(set(latent_dataframe.columns.values.tolist()))
        assert {'PATNO','EVENT_ID_DUR'}.issubset(set(future_times_dataframe.columns.values.tolist()))
        assert set(future_times_dataframe.PATNO.unique().tolist()).issubset(set(latent_dataframe.PATNO.unique().tolist()))
        future_patnos = future_times_dataframe.PATNO.values
        future_event_id_durs = future_times_dataframe.EVENT_ID_DUR.values
        future_latents = {'PATNO': future_patnos, 'EVENT_ID_DUR': future_event_id_durs}
        for latent_col in latent_column_names:
            future_latents[latent_col] = []
        for idx in range(len(future_patnos)):
            patno_latent_df = latent_dataframe.loc[latent_dataframe['PATNO']==future_patnos[idx]]
            for latent_col in latent_column_names:
                patno_latent_rate = patno_latent_df[latent_col+'_rate'].values[0]
                patno_latent_bias = patno_latent_df[latent_col+'_bias'].values[0]
                future_latents[latent_col].append(patno_latent_bias + patno_latent_rate*future_event_id_durs[idx])
        return pd.DataFrame(future_latents)
