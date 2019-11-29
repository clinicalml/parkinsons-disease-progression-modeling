import unittest, numpy as np, pandas as pd
from SurvivalOutcomeBuilder import SurvivalOutcomeBuilder
from SurvivalOutcomeCalculator import SurvivalOutcomeCalculator
from pandas.util.testing import assert_frame_equal

class TestSurvivalOutcomeDefinition(unittest.TestCase):

    '''
    Feature-level: Tests for identifying if a feature threshold satisfies the 2 criteria correctly:
    1. Less than 5% of patients observed to have crossed threshold under 2-visit persistence but not under 1-visit persistence -> Does not satisfy criteria
    2. More than 50% of patients have crossed threshold at 3 years -> Does not satisfy criteria
    3. Less than 50% of patients would have crossed threshold at 3 years with 1-visit persistence but not with 2-visit persistence and more than 5% of patients crossed threshold at end -> Satisfies criteria
    A/B. Test the above for both directions of feature thresholds
    '''
    def test_feature_less_than_5percent_above_dir(self): #1A
        feat_data = np.array([0,0, 1,0, 0,2,3, 0,3,0,3])
        patno_data = np.array([0,0, 1,1, 2,2,2, 3,3,3,3])
        event_id_data = np.array([0,1, 0,2, 0,2,4, 0,2,3,4])
        test_df = pd.DataFrame(np.vstack((patno_data, event_id_data, feat_data)).T, \
                               columns=['PATNO','EVENT_ID_DUR','test_feat'])
        feat_thresh = 3
        sob = SurvivalOutcomeBuilder(3, .5)
        self.assertFalse(sob.does_feature_threshold_satisfy_criteria(test_df, 'test_feat', feat_thresh, True))
    
    def test_feature_less_than_5percent_below_dir(self): #1B
        feat_data = np.array([1,2, 2,2, 3,1,2, 2,0,2,1])
        patno_data = np.array([0,0, 1,1, 2,2,2, 3,3,3,3])
        event_id_data = np.array([0,1, 0,2, 0,2,4, 0,2,3,4])
        test_df = pd.DataFrame(np.vstack((patno_data, event_id_data, feat_data)).T, \
                               columns=['PATNO','EVENT_ID_DUR','test_feat'])
        feat_thresh = 1
        sob = SurvivalOutcomeBuilder(3, .5)
        self.assertFalse(sob.does_feature_threshold_satisfy_criteria(test_df, 'test_feat', feat_thresh, False))

    def test_feature_more_than_50percent_above_dir(self): #2A
        feat_data = np.array([1,0, 2,2, 0,2,3, 2,2,1,2])
        patno_data = np.array([0,0, 1,1, 2,2,2, 3,3,3,3])
        event_id_data = np.array([0,1, 0,2, 0,2,4, 0,2,3,4])
        test_df = pd.DataFrame(np.vstack((patno_data, event_id_data, feat_data)).T, \
                               columns=['PATNO','EVENT_ID_DUR','test_feat'])
        feat_thresh = 2
        sob = SurvivalOutcomeBuilder(3, .5)
        self.assertFalse(sob.does_feature_threshold_satisfy_criteria(test_df, 'test_feat', feat_thresh, True))
    
    def test_feature_more_than_50percent_below_dir(self): #2B
        feat_data = np.array([1,0, 1,2, 2,1,1, 2,1,0,1])
        patno_data = np.array([0,0, 1,1, 2,2,2, 3,3,3,3])
        event_id_data = np.array([0,1, 0,2, 0,2,4, 0,2,3,4])
        test_df = pd.DataFrame(np.vstack((patno_data, event_id_data, feat_data)).T, \
                               columns=['PATNO','EVENT_ID_DUR','test_feat'])
        feat_thresh = 1
        sob = SurvivalOutcomeBuilder(3, .5)
        self.assertFalse(sob.does_feature_threshold_satisfy_criteria(test_df, 'test_feat', feat_thresh, False))
    
    def test_feature_satifies_2visit_persistence_above_dir(self): #3A
        feat_data = np.array([1,0, 1,2, 2,3,2, 3,1,2,1])
        patno_data = np.array([0,0, 1,1, 2,2,2, 3,3,3,3])
        event_id_data = np.array([0,1, 0,2, 0,2,4, 0,2,3,4])
        test_df = pd.DataFrame(np.vstack((patno_data, event_id_data, feat_data)).T, \
                               columns=['PATNO','EVENT_ID_DUR','test_feat'])
        feat_thresh = 2
        sob = SurvivalOutcomeBuilder(3, .5)
        self.assertTrue(sob.does_feature_threshold_satisfy_criteria(test_df, 'test_feat', feat_thresh, True))
    
    def test_feature_satifies_2visit_persistence_below_dir(self): #3B
        feat_data = np.array([0,2, 1,2, 2,0,1, 3,1,2,1])
        patno_data = np.array([0,0, 1,1, 2,2,2, 3,3,3,3])
        event_id_data = np.array([0,1, 0,2, 0,2,4, 0,2,3,4])
        test_df = pd.DataFrame(np.vstack((patno_data, event_id_data, feat_data)).T, \
                               columns=['PATNO','EVENT_ID_DUR','test_feat'])
        feat_thresh = 1
        sob = SurvivalOutcomeBuilder(3, .5)
        self.assertTrue(sob.does_feature_threshold_satisfy_criteria(test_df, 'test_feat', feat_thresh, False))

    '''
    Feature-level: Tests for identifying whether an event was observed or censored for a feature threshold for each patient and the observed/censored time:
    1. 1-visit persistence occurs before 2-visit persistence event time -> Observed
    2. Only 1-visit persistence, no 2-visit persistence -> Censored
        a. If last visit is at threshold, the time of the 2nd to last visit should be the censoring time.
        b. If last visit is not at threshold, the time of the last visit should be the censoring time.
    A/B. Test the above for both directions of feature thresholds
    '''
    def test_get_feature_time_event_df_above_dir(self): #1A + 2A
        feat_data = np.array([1,2, 2,3, 2,1,0, 2,1,2,2])
        patno_data = np.array([0,0, 1,1, 2,2,2, 3,3,3,3])
        event_id_data = np.array([0,1, 0,2, 0,2,4, 0,2,3,4])
        test_df = pd.DataFrame(np.vstack((patno_data, event_id_data, feat_data)).T, \
                               columns=['PATNO','EVENT_ID_DUR','test_feat'])
        output_patno_data = np.array([0,1,2,3])
        output_feat_time_data = np.array([0,0,4,3])
        output_feat_obs_data = np.array([0,1,0,1])
        output_df = pd.DataFrame(np.vstack((output_patno_data, output_feat_time_data, output_feat_obs_data)).T, \
                                 columns=['PATNO','test_feat_T','test_feat_E'])
        feat_thresh = 2
        sob = SurvivalOutcomeBuilder(3, .5)
        assert_frame_equal(sob._get_feature_time_event_df(test_df, 'test_feat', feat_thresh, True), output_df, \
                           check_dtype=False)
    
    def test_get_feature_time_event_df_below_dir(self): #1B + 2B
        feat_data = np.array([2,1, 1,0, 1,2,2, 1,2,1,1])
        patno_data = np.array([0,0, 1,1, 2,2,2, 3,3,3,3])
        event_id_data = np.array([0,1, 0,2, 0,2,4, 0,2,3,4])
        test_df = pd.DataFrame(np.vstack((patno_data, event_id_data, feat_data)).T, \
                               columns=['PATNO','EVENT_ID_DUR','test_feat'])
        output_patno_data = np.array([0,1,2,3])
        output_feat_time_data = np.array([0,0,4,3])
        output_feat_obs_data = np.array([0,1,0,1])
        output_df = pd.DataFrame(np.vstack((output_patno_data, output_feat_time_data, output_feat_obs_data)).T, \
                                 columns=['PATNO','test_feat_T','test_feat_E'])
        feat_thresh = 1
        sob = SurvivalOutcomeBuilder(3, .5)
        assert_frame_equal(sob._get_feature_time_event_df(test_df, 'test_feat', feat_thresh, False), output_df, \
                           check_dtype=False)

    '''
    Category-level/grouping-level: Tests for identifying whether at least N features in a category cross a threshold and observed/censored time:
    1. N features don't happen for the patient -> Censored at (N - # events observed)th censoring time
    2. N features happen for the patient where 1 distinct feature triggers the threshold -> Observed
    3. N features happen for the patient where at least 2 features trigger the threshold at the same time -> Observed
    '''
    def test_grouping_time_to_event(self):
        feat1_time_data = np.array([1,2,3,2,2])
        feat1_event_data = np.array([1,0,1,1,1])
        feat2_time_data = np.array([2,3,2,2,3])
        feat2_event_data = np.array([1,0,0,1,0])
        patno_data = np.array([0,1,2,3,4])
        test_df \
            = pd.DataFrame(np.vstack((patno_data, feat1_time_data, feat1_event_data, feat2_time_data, feat2_event_data)).T, \
                           columns=['PATNO','test_feat1_T','test_feat1_E','test_feat2_T','test_feat2_E'])
        output_patno_data = np.array([0,1,2,3,4])
        output_feat_time_data = np.array([2,3,2,2,3])
        output_feat_obs_data = np.array([1,0,0,1,0])
        output_df = pd.DataFrame(np.vstack((output_patno_data, output_feat_time_data, output_feat_obs_data)).T, \
                                 columns=['PATNO','group1_T','group1_E'])
        feat_grouping = {'test_feat1','test_feat2'}
        grouping_thresh = 2
        sob = SurvivalOutcomeBuilder(3, .5)
        assert_frame_equal(sob._get_grouping_time_event_df(test_df, 'group1', feat_grouping, grouping_thresh), output_df, \
                           check_dtype=False)

    '''
    Category-level/grouping-level/hybrid-level: Tests for identifying if a threshold N for the number of features required for a category satisfies the criteria:
    1. Less than 5% of patients have N features past their threshold at end -> Does not satisfy criteria
    2. More than 50% of patients have N features past their threshold at 3 years -> Does not satisfy criteria
    3. Less than 50% of patients have N features past their threshold at 3 years and more than 5% of patients crossed threshold at end -> Satisfies criteria
    '''
    def test_grouping_thresh_less_than_5percent(self):
        feat1_time_data = np.array([0,2,1,3])
        feat1_event_data = np.array([1,0,0,0])
        feat2_time_data = np.array([2,1,2,3])
        feat2_event_data = np.array([0,1,0,0])
        patno_data = np.array([0,1,2,3])
        test_df \
            = pd.DataFrame(np.vstack((patno_data, feat1_time_data, feat1_event_data, feat2_time_data, feat2_event_data)).T, \
                           columns=['PATNO','test_feat1_T','test_feat1_E','test_feat2_T','test_feat2_E'])
        feat_grouping = {'test_feat1','test_feat2'}
        grouping_thresh = 2
        sob = SurvivalOutcomeBuilder(3, .5)
        self.assertFalse(sob.does_grouping_threshold_satisfy_criteria(test_df, 'group1', feat_grouping, grouping_thresh))
    
    def test_grouping_thresh_more_than_50percent(self):
        feat1_time_data = np.array([0,2,1,3])
        feat1_event_data = np.array([1,0,0,0])
        feat2_time_data = np.array([2,1,2,3])
        feat2_event_data = np.array([1,1,1,0])
        patno_data = np.array([0,1,2,3])
        test_df \
            = pd.DataFrame(np.vstack((patno_data, feat1_time_data, feat1_event_data, feat2_time_data, feat2_event_data)).T, \
                           columns=['PATNO','test_feat1_T','test_feat1_E','test_feat2_T','test_feat2_E'])
        grouping_thresh = 1
        feat_grouping = {'test_feat1','test_feat2'}
        sob = SurvivalOutcomeBuilder(3, .5)
        self.assertFalse(sob.does_grouping_threshold_satisfy_criteria(test_df, 'group1', feat_grouping, grouping_thresh))
    
    def test_grouping_thresh_satisfies_criteria(self):
        feat1_time_data = np.array([3,2,4,3])
        feat1_event_data = np.array([0,1,1,0])
        feat2_time_data = np.array([2,1,3,2])
        feat2_event_data = np.array([0,1,1,1])
        patno_data = np.array([0,1,2,3])
        test_df \
            = pd.DataFrame(np.vstack((patno_data, feat1_time_data, feat1_event_data, feat2_time_data, feat2_event_data)).T, \
                           columns=['PATNO','test_feat1_T','test_feat1_E','test_feat2_T','test_feat2_E'])
        feat_grouping = {'test_feat1','test_feat2'}
        grouping_thresh = 2
        sob = SurvivalOutcomeBuilder(3, .5)
        self.assertTrue(sob.does_grouping_threshold_satisfy_criteria(test_df, 'group1', feat_grouping, grouping_thresh))
        
    '''
    Integration test for building outcome would be ideal but probably too hard to set up.
    '''
    
    '''
    Order of features: Tests for counting frequencies of orders of features:
    1. Patients with no feature events
    2. Patients with an incomplete set of feature events
    3. Patients with a full set of feature events
    4. Patients with ties up to full set of feature events, i.e. require 3 and have {A, B}, C or A, {B, C}
    5. Patients with ties at the grouping threshold, i.e. require 3 and have A, B, {C, D}
    '''
    def test_order_of_feature_counts(self):
        '''
        PATNO | T1 E1 | T2 E2 | T3 E3 | T4 E4 | str (3)
        ------------------------------------------------------
        0     | 1  0  | 3  0  | 2  0  | 1  0  | nan, nan, nan 
        1     | 1  1  | 3  0  | 2  1  | 1  0  | 1, 3, nan
        2     | 2  1  | 3  0  | 2  1  | 1  0  | (1, 3), nan
        3     | 1  1  | 2  0  | 3  1  | 2  1  | 1, 4, 3
        4     | 1  1  | 1  1  | 2  0  | 2  1  | (1, 2), 4
        5     | 3  0  | 3  1  | 3  1  | 1  1  | 4, (2, 3)
        6     | 2  1  | 1  1  | 3  1  | 3  1  | 2, 1, (3, 4)
        7     | 2  1  | 2  0  | 3  1  | 2  0  | 1, 3, nan
        8     | 1  1  | 2  0  | 3  1  | 3  0  | 1, 3, nan
        9     | 2  1  | 5  1  | 4  1  | 3  1  | 1, 4, 3
        10    | 2  1  | 2  0  | 3  0  | 2  0  | 1, nan, nan
        '''
        patno_data = np.array([0,1,2,3,4,5,6,7,8,9,10])
        feat1_time_data = np.array([1,1,2,1,1,3,2,2,1,2,2])
        feat1_event_data = np.array([0,1,1,1,1,0,1,1,1,1,1])
        feat2_time_data = np.array([3,3,3,2,1,3,1,2,2,3,2])
        feat2_event_data = np.array([0,0,0,0,1,1,1,0,0,0,0])
        feat3_time_data = np.array([2,2,2,3,2,3,3,3,3,4,3])
        feat3_event_data = np.array([0,1,1,1,0,1,1,1,1,1,0])
        feat4_time_data = np.array([1,1,1,2,2,1,3,2,3,3,2])
        feat4_event_data = np.array([0,0,0,1,1,1,1,0,0,1,0])
        test_df = pd.DataFrame(np.vstack((patno_data, feat1_time_data, feat1_event_data, feat2_time_data, feat2_event_data, \
                                          feat3_time_data, feat3_event_data, feat4_time_data, feat4_event_data)).T, \
                               columns = ['PATNO', 'feat1_T', 'feat1_E', 'feat2_T', 'feat2_E', \
                                          'feat3_T', 'feat3_E', 'feat4_T', 'feat4_E'])
        feat_grouping = {'feat1', 'feat2', 'feat3', 'feat4'}
        grouping_thresh = 3
        # feature thresholds and directions below don't matter since don't use in test
        all_feat_thresholds_directions = {'cat1': ({'feat1': ({'feat1': (1, True)}, 1), \
                                                    'feat2': ({'feat2': (1, True)}, 1), \
                                                    'feat3': ({'feat3': (1, True)}, 1), \
                                                    'feat4': ({'feat4': (1, True)}, 1)}, 3)}
        soc = SurvivalOutcomeCalculator(all_feat_thresholds_directions, 3)
        self.assertEqual(soc.group_lookup, {'feat1': 'cat1', 'feat2': 'cat1', 'feat3': 'cat1', 'feat4': 'cat1'})
        self.assertEqual(soc.feat_lookup, {'feat1': ('cat1', 'feat1'), 'feat2': ('cat1', 'feat2'), \
                                           'feat3': ('cat1', 'feat3'), 'feat4': ('cat1', 'feat4')})
        self.assertEqual(soc.get_patno_order_str(test_df.loc[test_df['PATNO']==0], 'cat1'), 'nan, nan, nan')
        self.assertEqual(soc.get_patno_order_str(test_df.loc[test_df['PATNO']==1], 'cat1'), 'feat1, feat3, nan')
        self.assertEqual(soc.get_patno_order_str(test_df.loc[test_df['PATNO']==2], 'cat1'), '(feat1, feat3), nan')
        self.assertEqual(soc.get_patno_order_str(test_df.loc[test_df['PATNO']==3], 'cat1'), 'feat1, feat4, feat3')
        self.assertEqual(soc.get_patno_order_str(test_df.loc[test_df['PATNO']==4], 'cat1'), '(feat1, feat2), feat4')
        self.assertEqual(soc.get_patno_order_str(test_df.loc[test_df['PATNO']==5], 'cat1'), 'feat4, (feat2, feat3)')
        self.assertEqual(soc.get_patno_order_str(test_df.loc[test_df['PATNO']==6], 'cat1'), 'feat2, feat1, (feat3, feat4)')
        self.assertEqual(soc.get_patno_order_str(test_df.loc[test_df['PATNO']==7], 'cat1'), 'feat1, feat3, nan')
        self.assertEqual(soc.get_patno_order_str(test_df.loc[test_df['PATNO']==8], 'cat1'), 'feat1, feat3, nan')
        self.assertEqual(soc.get_patno_order_str(test_df.loc[test_df['PATNO']==9], 'cat1'), 'feat1, feat4, feat3')
        self.assertEqual(soc.get_patno_order_str(test_df.loc[test_df['PATNO']==10], 'cat1'), 'feat1, nan, nan')
        expected_dict = dict()
        expected_dict = {'nan, nan, nan': 1, 'feat1, feat3, nan': 3, '(feat1, feat3), nan': 1, 'feat1, feat4, feat3': 2, \
                         '(feat1, feat2), feat4': 1, 'feat4, (feat2, feat3)': 1, 'feat2, feat1, (feat3, feat4)': 1, \
                         'feat1, nan, nan': 1}
        returned_dict, returned_str = soc.get_order_counts(test_df, 'cat1')
        self.assertEqual(returned_dict, expected_dict)
        expected_str = 'feat1, feat3, nan\t3\n' + 'feat1, feat4, feat3\t2\n' + '(feat1, feat2), feat4\t1\n' \
            + '(feat1, feat3), nan\t1\n' + 'feat1, nan, nan\t1\n' + 'feat2, feat1, (feat3, feat4)\t1\n'\
            + 'feat4, (feat2, feat3)\t1\n' + 'nan, nan, nan\t1\n'
        self.assertEqual(returned_str, expected_str)
        
    '''
    TODO: Need a test for get_time_event_df in SurvivalOutcomeCalculator
    2 categories: 1 category has 1 group with 1 feature, other category has 1 group with 2 features + 1 group with 1 feature
    Some features should be at least threshold and some at most
    '''
        
if __name__ == '__main__':
    unittest.main()
