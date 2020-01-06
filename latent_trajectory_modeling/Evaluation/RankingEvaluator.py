import numpy as np, pandas as pd, matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

'''
3 evaluation methods (using single latent factors):
1. Proportion of consecutive visits that obey ranking.
2. Concordance index: proportion of ordered visits that obey ranking.
3. Plot distribution of factor at each timepoint and overall distribution.
'''

class RankingEvaluator(object):
    
    def get_consec_visit_ranking(self, data, factor_col, plot_filepath=None, model_name=None, agg_backend=False):
        '''
        factor_col is a list of latent factor columns
        data: dataframe with PATNO, EVENT_ID_DUR, factor_col as columns
        plots share of consecutive visits for each patient that have latent factor in increasing order
        returns total share of consecutive visits that have latent factor in increasing order
        '''
        assert {'PATNO','EVENT_ID_DUR',factor_col}.issubset(set(data.columns.values.tolist()))
        assert len(data.dropna(subset=['PATNO','EVENT_ID_DUR',factor_col])) == len(data)
        share_obey_rank = []
        rank_len = []
        pat_list = data.PATNO.unique()
        for patno in pat_list:
            patno_df = data.loc[data['PATNO']==patno].sort_values(by=['EVENT_ID_DUR'])
            if len(patno_df) < 2:
                continue
            share_obey_rank.append(np.sum(np.where(patno_df[factor_col].diff() > 0, 1, 0))/float(len(patno_df)-1))
            rank_len.append(len(patno_df)-1)
        share_obey_rank = np.array(share_obey_rank)
        rank_len = np.array(rank_len)
        if agg_backend:
            plt.switch_backend('agg')
        plt.figure()
        plt.clf()
        plt.hist(share_obey_rank, bins=40)
        plt.xlabel('share of ' + factor_col + ' that obey time ranking')
        plt.ylabel('frequency')

        if model_name is not None:
            plt.title(model_name)
        if plot_filepath is not None:
            plt.savefig(plot_filepath)

        return np.sum(np.multiply(share_obey_rank, rank_len))/float(np.sum(rank_len))

    def get_concordance_index(self, df, factor_col):
        '''
        factor_col is a list of latent factor columns
        df: dataframe containing PATNO, EVENT_ID_DUR, factor_col as columns
        returns concordance index (proportion of correct orderings between all ordered pairs, not just consecutive visits)
        '''
        col_set = {'PATNO','EVENT_ID_DUR'} | set(factor_col)
        assert col_set.issubset(set(df.columns.values.tolist()))
        assert len(df.dropna(subset=['PATNO','EVENT_ID_DUR']+factor_col)) == len(df)
        num_correct_order_pairs = 0
        num_pairs = 0
        for patno in df.PATNO.unique():
            patno_df = df.loc[df['PATNO']==patno].sort_values(by=['EVENT_ID_DUR'])
            factor_vals = patno_df[factor_col].values
            num_timepoints = len(factor_vals)
            if num_timepoints < 2:
                continue
            for i in range(num_timepoints-1):
                for j in range(i+1,num_timepoints):
                    if all(np.greater(factor_vals[j] , factor_vals[i])):
                        num_correct_order_pairs += 1
            num_pairs += (num_timepoints)*(num_timepoints-1)/2
        return num_correct_order_pairs/float(num_pairs)
        
    def plot_latent_distributions_across_time(self, data, factor_col, plot_filepath, model_name, agg_backend=False):
        '''
        factor_col is a single latent factor here
        data: dataframe containg PATNO, EVENT_ID_DUR, factor_col
        model_name used only for plot title
        Makes several plots in a column sharing same x-,y-axes.
        1 plot is for all samples.
        Each row is a plot for a timepoint.
        '''
        assert {'PATNO','EVENT_ID_DUR',factor_col}.issubset(set(data.columns.values.tolist()))
        assert len(data.dropna(subset=['PATNO','EVENT_ID_DUR',factor_col])) == len(data)
        event_id_durs = [] # only keep timepoints with at least 50 samples
        for event_id_dur in data.EVENT_ID_DUR.unique():
            if len(data.loc[data['EVENT_ID_DUR']==event_id_dur]) >= .2*data.PATNO.nunique():
                event_id_durs.append(event_id_dur)
        event_id_durs.sort()   # sort timepoints
        if agg_backend:
            plt.switch_backend('agg')
        plt.clf()
        fig, ax = plt.subplots(nrows=len(event_id_durs)+1, ncols=1, sharex=True, figsize=(5,15)) # taller figures for x-axis label
        ax[0].hist(data[factor_col].values, bins=40)
        ax[0].set_xlabel('All timepoints')
        for time_idx in range(len(event_id_durs)):
            event_id_dur = event_id_durs[time_idx]
            ax[time_idx+1].hist(data.loc[data['EVENT_ID_DUR']==event_id_dur][factor_col].values, bins=40)
            ax[time_idx+1].set_xlabel('Timepoint ' + str(event_id_dur) + ' years')
        plt.tight_layout()
        plt.suptitle(model_name + ' ' + factor_col + ' distributions across time')
        plt.savefig(plot_filepath)
        