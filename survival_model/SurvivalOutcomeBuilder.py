import numpy as np, pandas as pd, os, sys, pickle
from SurvivalOutcomeHandler import SurvivalOutcomeHandler

class SurvivalOutcomeBuilder(SurvivalOutcomeHandler):
    
    def __init__(self, num_years, prop_pop):
        self.num_years = num_years
        self.prop_pop = prop_pop

    def does_feature_threshold_satisfy_criteria(self, df, feat_name, feat_thresh, feat_direction):
        '''
        df has PATNO, EVENT_ID_DUR, feat_name as columns
        '''
        assert {'PATNO', 'EVENT_ID_DUR', feat_name}.issubset(set(df.columns.values.tolist()))
        feat_time_event_df = self._get_feature_time_event_df(df, feat_name, feat_thresh, feat_direction)
        num_patnos = len(feat_time_event_df)
        num_event_patnos = len(feat_time_event_df.loc[feat_time_event_df[feat_name + '_E']==1])
        # At least p% of patients have data after n years
        feat_df = df.dropna(subset=['PATNO','EVENT_ID_DUR',feat_name])
        num_patnos_data_after_n_years = feat_df.loc[feat_df['EVENT_ID_DUR'] >= self.num_years].PATNO.nunique()
        if num_patnos_data_after_n_years/float(num_patnos) < self.prop_pop:
            return False
        # At least 5% of patients have event observed
        if num_event_patnos/float(num_patnos) < 0.05:
            return False
        # At most p% of patients have event within n years
        num_event_within_n_years = len(feat_time_event_df.loc[np.logical_and(feat_time_event_df[feat_name + '_E']==1, \
                                                                             feat_time_event_df[feat_name + '_T'] \
                                                                             < self.num_years)])
        if num_event_within_n_years/float(num_patnos) > self.prop_pop:
            return False
        return True

    def does_grouping_threshold_satisfy_criteria(self, df, grouping_name, feat_grouping, grouping_thresh):
        '''
        df has PATNO, feat1_T, feat1_E, feat2_T, feat2_E, etc. as columns
        feat_grouping: set of feature names
        '''
        assert 'PATNO' in set(df.columns.values.tolist())
        feat_T_list = [feat + '_T' for feat in feat_grouping]
        feat_E_list = [feat + '_E' for feat in feat_grouping]
        assert set(feat_T_list).issubset(set(df.columns.values.tolist()))
        assert set(feat_E_list).issubset(set(df.columns.values.tolist()))
        assert len(df) == df.PATNO.nunique()
        assert len(df) == len(df.dropna(subset=['PATNO']+feat_E_list+feat_T_list))
        grouping_time_event_df = self._get_grouping_time_event_df(df, grouping_name, feat_grouping, grouping_thresh)
        num_patnos = len(grouping_time_event_df)
        # At least 5% of patients have event observed
        num_event_patnos = len(grouping_time_event_df.loc[grouping_time_event_df[grouping_name + '_E']==1])
        if num_event_patnos/float(num_patnos) < 0.05:
            return False
        # At most p% of patients have event within n years 
        num_event_within_n_years \
            = len(grouping_time_event_df.loc[np.logical_and(grouping_time_event_df[grouping_name+'_E']==1, \
                                                            grouping_time_event_df[grouping_name+'_T'] < self.num_years)])
        if num_event_within_n_years/float(num_patnos) > self.prop_pop:
            return False
        return True    

    def get_feature_threshold(self, df, feat_name, feat_thresholds_to_try, feat_direction):
        '''
        df has PATNO, EVENT_ID_DUR, feat_name as columns
        '''
        assert {'PATNO', 'EVENT_ID_DUR', feat_name}.issubset(set(df.columns.values.tolist()))
        for threshold in feat_thresholds_to_try:
            if self.does_feature_threshold_satisfy_criteria(df, feat_name, threshold, feat_direction):
                return threshold
        return None

    def get_grouping_threshold(self, df, grouping_name, feat_grouping):
        '''
        df has PATNO, feat1_T, feat1_E, feat2_T, feat2_E, etc. as columns
        '''
        assert 'PATNO' in set(df.columns.values.tolist())
        feat_T_list = [feat + '_T' for feat in feat_grouping]
        feat_E_list = [feat + '_E' for feat in feat_grouping]
        assert set(feat_T_list).issubset(df.columns.values.tolist())
        assert set(feat_E_list).issubset(df.columns.values.tolist())
        assert len(df) == df.PATNO.nunique()
        assert len(df) == len(df.dropna(subset=['PATNO']+feat_E_list+feat_T_list))
        for threshold in range(1, len(feat_grouping)):
            if self.does_grouping_threshold_satisfy_criteria(df, grouping_name, feat_grouping, threshold):
                return threshold
        return None

    def get_all_feature_thresholds(self, df, feat_thresholds_to_try_directions):
        '''
        df has PATNO, EVENT_ID_DUR, feat1, feat2, etc. as columns
        feat_thresholds_to_try_directions is a dictionary feat -> (list of thresholds, direction)
        returns a dictionary of feat -> (thresh, direction)
        if a feature does not have a threshold that satisfies the criteria, it is removed
        '''
        assert {'PATNO', 'EVENT_ID_DUR'}.issubset(set(df.columns.values.tolist()))
        assert set(feat_thresholds_to_try_directions.keys()).issubset(set(df.columns.values.tolist()))
        feat_thresholds_directions = dict()
        for feat in feat_thresholds_to_try_directions.keys():
            feat_threshold = self.get_feature_threshold(df, feat, feat_thresholds_to_try_directions[feat][0], \
                                                        feat_thresholds_to_try_directions[feat][1])
            if feat_threshold is not None:
                feat_thresholds_directions[feat] = (feat_threshold, feat_thresholds_to_try_directions[feat][1])
        return feat_thresholds_directions

    def get_all_grouping_thresholds(self, df, category_thresholds_to_try_directions):
        '''
        df has PATNO, EVENT_ID_DUR, feat1, feat2, etc. as columns
        category_thresholds_to_try_directions is a 2-level dictionary: 
            grouping_name -> (feat_name -> (list of thresholds, direction)
        returns 1. 2-level dictionary for grouping_name -> (feat_name -> (feat_thresh, feat_dir), grouping_thresh)
                2. dataframe with PATNO, feat1_T, feat1_E, feat2_T, feat2_E, etc., 
                                  grouping1_T, grouping1_E, grouping2_T, grouping2_E, etc. as columns
        '''
        assert {'PATNO', 'EVENT_ID_DUR'}.issubset(set(df.columns.values.tolist()))
        for grouping in category_thresholds_to_try_directions.keys():
            assert set(category_thresholds_to_try_directions[grouping].keys()).issubset(set(df.columns.values.tolist()))
        all_feat_time_event_df = pd.DataFrame(df.PATNO.unique(), columns=['PATNO'])
        category_thresholds_directions = dict()
        for grouping in category_thresholds_to_try_directions.keys():
            grouping_thresholds_to_try_directions = category_thresholds_to_try_directions[grouping]
            grouping_thresholds_directions = self.get_all_feature_thresholds(df, grouping_thresholds_to_try_directions)
            if len(grouping_thresholds_directions) == 0:
                continue
            for feat in grouping_thresholds_directions.keys():
                feature_time_event_df = self._get_feature_time_event_df(df, feat, grouping_thresholds_directions[feat][0], \
                                                                        grouping_thresholds_directions[feat][1])
                all_feat_time_event_df = all_feat_time_event_df.merge(feature_time_event_df, on=['PATNO'], how='left', \
                                                                      validate='one_to_one')
            if len(grouping_thresholds_directions) == 1:
                # threshold of 1 will always work
                grouping_threshold = 1
            else:
                grouping_threshold = self.get_grouping_threshold(all_feat_time_event_df, grouping, \
                                                                 grouping_thresholds_directions.keys())
            if grouping_threshold is not None:
                category_thresholds_directions[grouping] = (grouping_thresholds_directions, grouping_threshold)
                if not(len(grouping_thresholds_directions) == 1 and grouping_thresholds_directions.keys()[0] == grouping):
                    grouping_time_event_df = self._get_grouping_time_event_df(all_feat_time_event_df, grouping, \
                                                                              grouping_thresholds_directions.keys(), \
                                                                              grouping_threshold)
                    all_feat_time_event_df = all_feat_time_event_df.merge(grouping_time_event_df, on=['PATNO'], how='left', \
                                                                          validate='one_to_one')
        return category_thresholds_directions, all_feat_time_event_df.reset_index(drop=True)
    
    def get_all_thresholds(self, df, all_feat_thresholds_to_try_directions):
        '''
        all_feat_thresholds_to_try_directions is a 3-level dictionary: category -> grouping_name -> feat_name -> (list_of_thresholds, direction)
        returns 1. a 3-level dictionary and a total threshold:
                    Level 1: category -> (groupings, category threshold)
                    Level 2: grouping_name -> (feats, grouping threshold)
                    Level 3: feat_name -> (feature threshold, feature direction)
                2. Threshold for number of categories required for hybrid
                3. a dataframe with PATNO, feat1_T, feat1_E, feat2_T, feat2_E, etc., 
                                    grouping1_T, grouping1_E, grouping2_T, grouping2_E, etc., hybrid_T, hybrid_E 
        returns None if infeasible with criteria
        '''
        for category in all_feat_thresholds_to_try_directions.keys():
            for grouping in all_feat_thresholds_to_try_directions[category].keys():
                grouping_feats = set(all_feat_thresholds_to_try_directions[category][grouping].keys())
                assert grouping_feats.issubset(set(df.columns.values.tolist()))
        # TODO: when have time, pull out a method covering code in for loop body because duplicated for hybrid outcome
        all_feat_thresholds_directions = dict()
        all_feat_event_time_df = pd.DataFrame(df.PATNO.unique(), columns=['PATNO'])
        for category in all_feat_thresholds_to_try_directions.keys():
            category_thresholds_to_try_directions = all_feat_thresholds_to_try_directions[category]
            category_thresholds_directions, category_feat_time_event_df \
                = self.get_all_grouping_thresholds(df, category_thresholds_to_try_directions)
            all_feat_event_time_df = all_feat_event_time_df.merge(category_feat_time_event_df, on=['PATNO'], how='left', \
                                                                  validate='one_to_one')
            if len(category_thresholds_directions) == 0:
                continue
            if len(category_thresholds_directions) == 1:
                # threshold of 1 will always work
                category_threshold = 1
            else:
                category_threshold = self.get_grouping_threshold(all_feat_event_time_df, category, \
                                                                 category_thresholds_directions.keys())
            if category_threshold is not None:
                all_feat_thresholds_directions[category] = (category_thresholds_directions, category_threshold)
                if not(len(category_thresholds_directions) == 1 and category_thresholds_directions.keys()[0] == category):
                    category_event_time_df = self._get_grouping_time_event_df(all_feat_event_time_df, category, \
                                                                              category_thresholds_directions.keys(), \
                                                                              category_threshold)
                    all_feat_event_time_df = all_feat_event_time_df.merge(category_event_time_df, on=['PATNO'], how='left', \
                                                                          validate='one_to_one')
        if len(all_feat_thresholds_directions) == 0:
            return None
        if len(all_feat_thresholds_directions) == 1:
            hybrid_threshold = 1
        else:
            hybrid_threshold = self.get_grouping_threshold(all_feat_event_time_df, 'hybrid', \
                                                           all_feat_thresholds_directions.keys())
        if hybrid_threshold is not None:
            # to avoid confusion hybrid shouldn't already be a column name in use
            assert 'hybrid_T' not in set(all_feat_event_time_df.columns.values.tolist())
            assert 'hybrid_E' not in set(all_feat_event_time_df.columns.values.tolist())
            hybrid_event_time_df = self._get_grouping_time_event_df(all_feat_event_time_df, 'hybrid', \
                                                                    all_feat_thresholds_directions.keys(), hybrid_threshold)
            all_feat_event_time_df = all_feat_event_time_df.merge(hybrid_event_time_df, on=['PATNO'], how='left', \
                                                                  validate='one_to_one')
            if 'Motor' in all_feat_thresholds_directions.keys():
                all_feat_event_time_df['hybrid_requiremotor_E'] = np.where(all_feat_event_time_df['Motor_E'] == 0, 0, \
                                                                           all_feat_event_time_df['hybrid_E'])
                '''
                Motor | Hybrid | Time
                -------------------------
                Obs   | Obs    | Later
                Obs   | Cens   | Hybrid
                Cens  | Obs    | Motor
                Cens  | Cens   | Earlier
                '''
                all_feat_event_time_df['hybrid_requiremotor_T'] \
                    = np.where(all_feat_event_time_df['Motor_E'] == 1, \
                               np.where(all_feat_event_time_df['hybrid_E'] == 1, \
                                        all_feat_event_time_df[['Motor_T', 'hybrid_T']].max(axis=1), \
                                        all_feat_event_time_df['hybrid_T']), \
                               np.where(all_feat_event_time_df['hybrid_E'] == 1, \
                                        all_feat_event_time_df['Motor_T'], \
                                        all_feat_event_time_df[['Motor_T', 'hybrid_T']].min(axis=1)))
            return all_feat_thresholds_directions, hybrid_threshold, all_feat_event_time_df.reset_index(drop=True)
        return None # infeasible
    