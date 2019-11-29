import numpy as np, pandas as pd, os, sys, pickle

class SurvivalOutcomeHandler(object):
    
    '''
    This class should only be instantiated via its inherited classes and its methods only called internally as helper methods.
    '''
    
    def _get_feature_time_event_df(self, df, feat, feat_thresh, feat_direction):
        '''
        df has PATNO, EVENT_ID_DUR, feat_name as columns
        returns dataframe with PATNO, feat_name_T, feat_name_E as columns
        '''
        assert {'PATNO', 'EVENT_ID_DUR', feat}.issubset(set(df.columns.values.tolist()))
        feat_df = df[['PATNO','EVENT_ID_DUR',feat]].dropna().sort_values(by=['PATNO','EVENT_ID_DUR'])
        if feat_direction:
            feat_df[feat + '_thresh'] = np.where(feat_df[feat] >= feat_thresh, 1, 0)
        else:
            feat_df[feat + '_thresh'] = np.where(feat_df[feat] <= feat_thresh, 1, 0)
        feat_te_df = df[['PATNO']].drop_duplicates()
        feat_te_df[feat+'_T'] = float('NaN')
        feat_te_df[feat+'_E'] = 0
        for row_idx in range(len(feat_te_df)):
            patno = feat_te_df.iloc[[row_idx]].PATNO.values[0]
            if patno not in set(feat_df.PATNO.values.tolist()):
                feat_te_df.iloc[row_idx, feat_te_df.columns.get_loc(feat+'_T')] = 0 # never observed, so censored at 0
                continue
            patno_feat_df = feat_df.loc[feat_df['PATNO']==patno]
            if len(patno_feat_df) == 1:
                # 2 possible cases for censored:
                # a. If the single visit is at threshold, censoring time should be 0.
                # b. If the single visit is not at threshold, censoring time should be visit time
                if patno_feat_df[feat + '_thresh'].values[0] == 1:
                    feat_te_df.iloc[row_idx, feat_te_df.columns.get_loc(feat+'_T')] = 0
                else:
                    feat_te_df.iloc[row_idx, feat_te_df.columns.get_loc(feat+'_T')] \
                        = patno_feat_df.EVENT_ID_DUR.values[-1]
                continue
            patno_feat_2visit_thresh_idxs \
                = np.nonzero(np.where((patno_feat_df[feat + '_thresh'].values[:-1] \
                                       + patno_feat_df[feat + '_thresh'].values[1:]).flatten() == 2, 1, 0))[0]
            if len(patno_feat_2visit_thresh_idxs) > 0:
                # observed
                feat_te_df.iloc[row_idx, feat_te_df.columns.get_loc(feat+'_E')] = 1
                feat_te_df.iloc[row_idx, feat_te_df.columns.get_loc(feat+'_T')] \
                    = patno_feat_df.EVENT_ID_DUR.values[patno_feat_2visit_thresh_idxs[0]]
            else:
                # 2 possible cases for censored:
                # a. If last visit is at threshold, the time of the 2nd to last visit should be the censoring time.
                # b. If last visit is not at threshold, the time of the last visit should be the censoring time.
                if patno_feat_df[feat + '_thresh'].values[-1] == 1:
                    feat_te_df.iloc[row_idx, feat_te_df.columns.get_loc(feat+'_T')] = patno_feat_df.EVENT_ID_DUR.values[-2]
                else:
                    feat_te_df.iloc[row_idx, feat_te_df.columns.get_loc(feat+'_T')] = patno_feat_df.EVENT_ID_DUR.values[-1]
        return feat_te_df.reset_index(drop=True)

    def _get_grouping_time_event_df(self, df, grouping_name, feat_grouping, grouping_thresh):
        '''
        df has PATNO, feat1_T, feat1_E, feat2_T, feat2_E, etc. as columns
        feat_grouping: set of feature names
        returns df with PATNO, grouping_T, grouping_E as columns
        '''
        assert grouping_thresh > 0
        feats = list(feat_grouping)
        feat_T_list = np.array([feat + '_T' for feat in feats])
        feat_E_list = np.array([feat + '_E' for feat in feats])
        assert 'PATNO' in df.columns.values
        assert set(feat_T_list).issubset(set(df.columns.values.tolist()))
        assert set(feat_E_list).issubset(set(df.columns.values.tolist()))
        if len(feat_grouping) == 1:
            single_feat_name = list(feat_grouping)[0]
            cat_thresh_df = df[['PATNO',single_feat_name + '_T', single_feat_name + '_E']]
            if single_feat_name == grouping_name:
                return cat_thresh_df
            return cat_thresh_df.rename(columns={single_feat_name + '_T': grouping_name + '_T', \
                                                 single_feat_name + '_E': grouping_name + '_E'})
        cat_thresh_df = df[['PATNO']]
        cat_thresh_df[grouping_name+'_T'] = float('NaN')
        cat_thresh_df[grouping_name+'_E'] = 0
        for row_idx in range(len(cat_thresh_df)):
            patno_df = df.iloc[[row_idx]]
            event_feat_T_list = feat_T_list[np.nonzero(patno_df[feat_E_list].values.flatten())[0]]
            if len(event_feat_T_list) >= grouping_thresh:
                event_feat_Ts = patno_df[event_feat_T_list].values.flatten()
                cat_thresh_df.iloc[row_idx, cat_thresh_df.columns.get_loc(grouping_name+'_T')] \
                    = event_feat_Ts[np.argpartition(event_feat_Ts, grouping_thresh - 1)[grouping_thresh - 1]]
                cat_thresh_df.iloc[row_idx, cat_thresh_df.columns.get_loc(grouping_name+'_E')] = 1
            else:
                # if censored, then time should be (grouping_thresh - # observed)th censoring time
                censored_feat_T_list = feat_T_list[np.nonzero(np.where(patno_df[feat_E_list].values.flatten()==0,1,0))[0]]
                censored_feat_Ts = patno_df[censored_feat_T_list].values.flatten()
                censor_time_idx = grouping_thresh - len(event_feat_T_list) - 1
                cat_thresh_df.iloc[row_idx, cat_thresh_df.columns.get_loc(grouping_name+'_T')] \
                    = np.partition(censored_feat_Ts, censor_time_idx)[censor_time_idx]
        return cat_thresh_df.reset_index(drop=True)
