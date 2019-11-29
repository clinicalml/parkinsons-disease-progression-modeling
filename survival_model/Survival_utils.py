import numpy as np, pandas as pd
from lifelines.statistics import logrank_test

def handle_zero_times(train_df, test_df, outcome_col):
    # for patients who are observed at 0, store test set for later prediction, drop for training, keep for valid
    # for patients who are censored at 0, make the censoring time 0.01 for training, keep for test + valid
    zero_duration_obs_test_patients = test_df.loc[np.logical_and(test_df[outcome_col + '_E']==1, \
                                                                 test_df[outcome_col + '_T']==0)].PATNO.values
    test_df = test_df.loc[~test_df['PATNO'].isin(zero_duration_obs_test_patients)]
    train_df[outcome_col + '_T'] = np.where(np.logical_and(train_df[outcome_col + '_E']==0, \
                                                           train_df[outcome_col + '_T']==0), \
                                            0.01, train_df[outcome_col + '_T'])
    train_df = train_df.loc[train_df[outcome_col + '_T']!=0]
    # insert 0 as prediction output for patients who were observed at 0
    zero_duration_obs_test_df = pd.DataFrame({'PATNO': zero_duration_obs_test_patients, \
                                              outcome_col + '_T_pred': \
                                              np.zeros(len(zero_duration_obs_test_patients))})
    return train_df, test_df, zero_duration_obs_test_df

def get_stratifying_feats(df, pval_thresh=0.05):
    outcomes = ['hybrid_requiremotor', 'Motor', 'Cognitive', 'Autonomic', 'Sleep', 'Psychiatric']
    outcome_stratifying_feats = dict()
    for outcome in outcomes:
        outcome_stratifying_feats[outcome] = set()
    for feat in df.columns.values[1:]:
        if df[feat].nunique() == 2:
            first_strata_df = df.loc[df[feat]==df[feat].min()]
            second_strata_df = df.loc[df[feat]==df[feat].max()]
            if len(first_strata_df) < 10 or len(second_strata_df) < 10:
                continue
            for outcome in outcomes:
                results = logrank_test(first_strata_df[outcome + '_T'], second_strata_df[outcome + '_T'], \
                                       first_strata_df[outcome + '_E'], second_strata_df[outcome + '_E'])
                if results.p_value <= pval_thresh:
                    outcome_stratifying_feats[outcome].add(feat)
        else:
            percentiles = [0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75]
            split_vals = []
            for percentile in percentiles:
                split_vals.append(df[feat].quantile(percentile))
            split_vals = np.array(split_vals).unique()
            for split_val in split_vals:
                first_strata_df = df.loc[df[feat]<=split_val]
                second_strata_df = df.loc[df[feat]>split_val]
                if len(first_strata_df) < 10 or len(second_strata_df) < 10:
                    continue
                for outcome in outcomes:
                    results = logrank_test(first_strata_df[outcome + '_T'], second_strata_df[outcome + '_T'], \
                                           first_strata_df[outcome + '_E'], second_strata_df[outcome + '_E'])
                    if results.p_value <= pval_thresh:
                        outcome_stratifying_feats[outcome].add(feat)
    return outcome_stratifying_feats

def get_hybrid_time(df):
    # recalculate hybrid using predictions
    assert {'Motor_T_pred','Cognitive_T_pred','Autonomic_T_pred','Psychiatric_T_pred',\
            'Sleep_T_pred'}.issubset(set(df.columns.values.tolist()))
    hybrid_times = np.empty(len(df))
    for row_idx in range(len(df)):
        times = df[['Cognitive_T_pred','Autonomic_T_pred','Psychiatric_T_pred','Sleep_T_pred']].values[row_idx]
        time_idxs = np.argsort(times)
        second_time = time_idxs[1]
        hybrid_times[row_idx] = max(second_time, df[['Motor_T_pred']].values[row_idx])
    df['hybrid_built_T_pred'] = hybrid_times
    return df

def get_patno_folds(patnos):
    np.random.seed(29033)
    np.random.shuffle(patnos)
    split20_idx = int(0.2*len(patnos))
    split40_idx = int(0.4*len(patnos))
    split60_idx = int(0.6*len(patnos))
    split80_idx = int(0.8*len(patnos))
    patno_folds = [{'train': patnos[:split60_idx], 'valid': patnos[split60_idx:split80_idx], \
                    'test': patnos[split80_idx:]}, \
                   {'train': patnos[split20_idx:split80_idx], 'valid': patnos[split80_idx:], \
                    'test': patnos[:split20_idx]}, \
                   {'train': patnos[split40_idx:], 'valid': patnos[:split20_idx], \
                    'test': patnos[split20_idx:split40_idx]}, \
                   {'train': np.concatenate((patnos[split60_idx:], patnos[:split20_idx])), \
                    'valid': patnos[split20_idx:split40_idx], 'test': patnos[split40_idx:split60_idx]}, \
                   {'train': np.concatenate((patnos[split80_idx:], patnos[:split40_idx])), \
                    'valid': patnos[split40_idx:split60_idx], 'test': patnos[split60_idx:split80_idx]}]
    return patno_folds
