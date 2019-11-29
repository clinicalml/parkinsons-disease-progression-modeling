import numpy as np, pandas as pd, os, sys, pickle, matplotlib, math
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from SurvivalOutcomeHandler import SurvivalOutcomeHandler
from lifelines import KaplanMeierFitter

class SurvivalOutcomeCalculator(SurvivalOutcomeHandler):
    
    def __init__(self, all_feat_thresholds_directions, hybrid_threshold, totals_or_questions, human_readable_dict, min_max_dict):
        '''
        all_feat_thresholds_directions is a 3-level dictionary
            category -> (group_dict, category threshold)
            group_dict: group -> (feat_dict, group threshold)
            feat_dict: feat -> (threshold, direction)
        all group and feature names must be unique across all categories
        all feature names must be in human_readable_dict
        totals_or_questions: totals, subtotals, or questions
        last 3 are used for printing purposes
        '''
        assert hybrid_threshold > 0
        assert 'hybrid' not in set(all_feat_thresholds_directions.keys())
        assert 'hybrid_requiremotor' not in set(all_feat_thresholds_directions.keys())
        for category in all_feat_thresholds_directions.keys():
            assert len(all_feat_thresholds_directions[category][0]) != 0
            assert isinstance(all_feat_thresholds_directions[category][1], int)
            assert all_feat_thresholds_directions[category][1] > 0
            assert 'hybrid' not in set(all_feat_thresholds_directions[category][0].keys())
            assert 'hybrid_requiremotor' not in set(all_feat_thresholds_directions[category][0].keys())
            for grouping in all_feat_thresholds_directions[category][0].keys():
                assert len(all_feat_thresholds_directions[category][0][grouping][0]) != 0
                assert isinstance(all_feat_thresholds_directions[category][0][grouping][1], int)
                assert all_feat_thresholds_directions[category][0][grouping][1] > 0
                assert 'hybrid' not in set(all_feat_thresholds_directions[category][0][grouping][0].keys())
                assert 'hybrid_requiremotor' not in set(all_feat_thresholds_directions[category][0][grouping][0].keys())
                for feature in all_feat_thresholds_directions[category][0][grouping][0]:
                    assert len(all_feat_thresholds_directions[category][0][grouping][0][feature]) == 2
                    assert feature in human_readable_dict.keys()
                    assert feature in min_max_dict.keys()
        assert totals_or_questions in {'totals', 'subtotals', 'questions'}
        self.hybrid_threshold = hybrid_threshold
        self.all_feat_thresholds_directions = all_feat_thresholds_directions
        self.totals_or_questions = totals_or_questions
        self.human_readable_dict = human_readable_dict
        self.min_max_dict = min_max_dict
        '''
        Store a 2 dictionaries for easier look-up:
            feat -> (category, group)
            group -> category
        '''
        self.feat_lookup = dict()
        self.group_lookup = dict()
        for category in self.all_feat_thresholds_directions.keys():
            for grouping in self.all_feat_thresholds_directions[category][0].keys():
                self.group_lookup[grouping] = category
                for feature in self.all_feat_thresholds_directions[category][0][grouping][0].keys():
                    self.feat_lookup[feature] = (category, grouping)
    
    def get_feature_time_event_df(self, df, feat_name):
        # only take feature name since feature threshold and direction can be found in stored dictionary
        assert feat_name in self.feat_lookup.keys()
        category, group = self.feat_lookup[feat_name]
        feat_thresh, feat_dir = self.all_feat_thresholds_directions[category][0][group][0][feat_name]
        return self._get_feature_time_event_df(df, feat_name, feat_thresh, feat_dir)
        
    def get_grouping_time_event_df(self, df, grouping_name):
        # take only grouping name since grouping features and grouping threshold are stored
        assert grouping_name in self.group_lookup.keys()
        category = self.group_lookup[grouping_name]
        feat_grouping = self.all_feat_thresholds_directions[category][0][grouping_name][0].keys()
        grouping_thresh = self.all_feat_thresholds_directions[category][0][grouping_name][1]
        return self._get_grouping_time_event_df(df, grouping_name, feat_grouping, grouping_thresh)
    
    def _order_dict_to_str(self, order_counts_dict, grouping_name, num_events, total_num_patnos, \
                           num_to_print=10):
        '''
        this is a helper method that should only be called inside get_order_counts below, hence it starts with _
        convert dictionary to a pair of lists for easy sorting
        ties are broken alphabetically, only top 10 are printed even if more are tied with the last rank
        string returned is a LaTeX table
        (2-column table)
        \begin{table}
            \centering
            \begin{tabular}{|p{10cm}|c|}
                \hline
                Order (first n events) & Count (out of total #) \\
                \hline
                order1 & count \\
                \hline
                order2 & count \\
                \hline
                ...
                \hline
            \end{tabular}
            \caption{\textbf{Order} of features for \textbf{grouping_name} using \textbf{totals/questions}. Ties are denoted within brackets.}
            \label{surv_outcome:table:grouping_name_orders_totals/questions}
        \end{table}
        TODO: unit test for this method should also be modified to check for LaTeX table
        '''
        assert num_events > 0
        assert total_num_patnos > 0
        assert num_to_print > 0
        # sort alphabetically so ordinal will break in this order but reversed since want higher rank for large elements
        order_strs = order_counts_dict.keys()
        order_strs.sort()
        order_strs = np.array(order_strs[::-1])
        order_str_counts = []
        for order_str in order_strs:
            order_str_counts.append(order_counts_dict[order_str])
        order_str_counts = np.array(order_str_counts)
        num_orders = min(num_to_print, len(order_str_counts))
        sorted_idxs = rankdata(order_str_counts, method='ordinal').tolist()
        top10_idxs = np.argpartition(sorted_idxs, int(-1*num_orders))[int(-1*num_orders):]
        top10_counts = order_str_counts[top10_idxs]
        top10_strs = order_strs[top10_idxs]
        top10_idxs_ordered = np.array(np.argsort(order_str_counts[top10_idxs]).tolist()[::-1])
        order_str_output = '\\begin{table}\n' + '\t\\centering\n' + '\t\\begin{tabular}{|p{10cm}|c|}\n' \
            + '\t\t\\hline\n' + '\t\tOrder (first ' + str(num_events) + ') & Count (out of ' + str(total_num_patnos) \
            + ') \\\\\n\t\t\hline\n'
        for order_str in top10_strs[top10_idxs_ordered]:
            order_str_output += '\t\t' + order_str.replace('_', '\\_') + ' & ' + str(order_counts_dict[order_str]) \
                + '\\\\\n\t\t\\hline\n'
        order_str_output += '\t\\end{tabular}\n' + '\t\\caption{\\textbf{Order} of features for \\textbf{' + grouping_name \
            + '} using \\textbf{' + self.totals_or_questions + '}. Ties are denoted with brackets.}\n' + '\t\\label{surv_outcome:table:' + grouping_name \
            + '_orders_' + self.totals_or_questions + '}\n' + '\\end{table}\n'
        return order_str_output
    
    def _get_patno_order_str(self, patno_df, feat_grouping, grouping_thresh):
        '''
        this is a helper method to avoid looking up the last 2 parameters excessively
        patno_df has PATNO, feat1_T, feat1_E, feat2_T, feat2_E, etc. as columns
        returns string representing order of events for this patient
        only attached at end if number of events for patient is less than grouping_thresh
        None if no events
        if feat1 and feat2 are tied and grouping_thresh is 3, any of the following are valid configurations:
            (feat1, feat2), feat3
            feat3, (feat1, feat2)
            feat3, feat4, (feat1, feat2)
        '''
        assert len(patno_df) == 1
        feat_list = list(feat_grouping)
        feat_list.sort() # alphabetical so ties will be broken in same order every time
        feat_list = np.array(feat_list)
        feat_T_list = np.array([feat + '_T' for feat in feat_list])
        feat_E_list = np.array([feat + '_E' for feat in feat_list])
        assert set(feat_T_list).issubset(set(patno_df.columns.values.tolist()))
        assert set(feat_E_list).issubset(set(patno_df.columns.values.tolist()))
        assert isinstance(grouping_thresh, int)
        assert grouping_thresh > 0
        event_feat_idxs = np.nonzero(patno_df[feat_E_list].values.flatten())[0]
        event_feat_T_list = feat_T_list[event_feat_idxs]
        event_feat_list = feat_list[event_feat_idxs]
        if len(event_feat_T_list) == 0:
            return 'None'
        event_feat_Ts = patno_df[event_feat_T_list].values.flatten()
        event_ranks = rankdata(event_feat_Ts, method='min')
        rank = 1
        order_str = ''
        while rank <= min(len(event_feat_T_list), grouping_thresh):
            rank_idxs = np.nonzero(np.where(event_ranks == rank, 1, 0))[0]
            if len(rank_idxs) == 1:
                feat = event_feat_list[rank_idxs[0]]
                if feat in self.human_readable_dict.keys(): # should only be the case for feature-level, not group/category-level
                    order_str += self.human_readable_dict[feat] + ', '
                else:
                    order_str += feat + ', '
            else:
                tied_events = event_feat_list[rank_idxs]
                human_readable_tied_events = []
                for feat in tied_events:
                    if feat in self.human_readable_dict.keys():
                        human_readable_tied_events.append(self.human_readable_dict[feat])
                    else:
                        human_readable_tied_events.append(feat)
                order_str += '\\{' + ', '.join(human_readable_tied_events) + '\\}, '
            rank += len(rank_idxs)
        if rank <= grouping_thresh:
            return order_str[:-2] + ' only'
        return order_str[:-2] # remove ', ' at end
    
    def get_patno_order_str(self, patno_df, group_name):
        '''
        this is the outward-facing equivalent of above
        '''
        if group_name in {'hybrid', 'hybrid_requiremotor'}:
            grouping_thresh = self.hybrid_threshold
            feat_grouping = self.all_feat_thresholds_directions.keys()
        elif group_name in self.all_feat_thresholds_directions.keys():
            grouping_thresh = self.all_feat_thresholds_directions[group_name][1]
            feat_grouping = self.all_feat_thresholds_directions[group_name][0].keys()
        else:
            assert group_name in self.group_lookup.keys()
            category = self.group_lookup[group_name]
            grouping_thresh = self.all_feat_thresholds_directions[category][0][group_name][1]
            feat_grouping = self.all_feat_thresholds_directions[category][0][group_name][0].keys()
        return self._get_patno_order_str(patno_df, feat_grouping, grouping_thresh)
    
    def get_order_counts(self, df, group_name):
        '''
        group_name can be 'hybrid', 'hybrid_requiremotor', category name, or group name
        df has feat1_T, feat1_E, feat2_T, feat2_E, etc. as columns, 
            where feat1, feat2, etc. are the level immediately below group_name
        returns a dictionary: string of order of features -> counts in df
        returns string representing top 10 from dictionary
        '''
        if group_name in {'hybrid', 'hybrid_requiremotor'}:
            grouping_thresh = self.hybrid_threshold
            feat_grouping = self.all_feat_thresholds_directions.keys()
        elif group_name in self.all_feat_thresholds_directions.keys():
            grouping_thresh = self.all_feat_thresholds_directions[group_name][1]
            feat_grouping = self.all_feat_thresholds_directions[group_name][0].keys()
        else:
            assert group_name in self.group_lookup.keys()
            category = self.group_lookup[group_name]
            grouping_thresh = self.all_feat_thresholds_directions[category][0][group_name][1]
            feat_grouping = self.all_feat_thresholds_directions[category][0][group_name][0].keys()
        
        feat_list = list(feat_grouping)
        feat_list.sort() # alphabetical so ties will be broken in same order every time
        feat_T_list = [feat + '_T' for feat in feat_list]
        feat_E_list = [feat + '_E' for feat in feat_list]
        assert set(feat_T_list).issubset(set(df.columns.values.tolist()))
        assert set(feat_E_list).issubset(set(df.columns.values.tolist()))
        assert isinstance(grouping_thresh, int)
        assert grouping_thresh > 0
        order_of_events = dict()
        for row_idx in range(len(df)):
            patno_df = df.iloc[[row_idx]]
            order_str = self._get_patno_order_str(patno_df, feat_grouping, grouping_thresh)
            if order_str in order_of_events.keys():
                order_of_events[order_str] = order_of_events[order_str] + 1
            else:
                order_of_events[order_str] = 1
        order_of_events_str = self._order_dict_to_str(order_of_events, group_name, grouping_thresh, df.PATNO.nunique())
        return order_of_events, order_of_events_str

    def get_time_event_df(self, df):
        '''
        df has PATNO, EVENT_ID_DUR, feat1, feat2, etc. as columns
        returns df with PATNO, feat1_T, feat1_E, feat2_T, feat2_E, etc., 
                        grouping1_T, grouping1_E, grouping2_T, grouping2_E, etc., hybrid_T, hybrid_E as columns
        '''
        assert {'PATNO', 'EVENT_ID_DUR'}.issubset(set(df.columns.values.tolist()))
        all_feat_time_event_df = pd.DataFrame(df.PATNO.unique(), columns=['PATNO'])
        for category in self.all_feat_thresholds_directions.keys():
            category_thresholds_directions = self.all_feat_thresholds_directions[category][0]
            for grouping in category_thresholds_directions.keys():
                grouping_thresholds_directions = category_thresholds_directions[grouping][0]
                for feat in grouping_thresholds_directions.keys():
                    assert feat in set(df.columns.values.tolist())
                    feat_time_event_df = self._get_feature_time_event_df(df, feat, grouping_thresholds_directions[feat][0], \
                                                                         grouping_thresholds_directions[feat][1])
                    all_feat_time_event_df = all_feat_time_event_df.merge(feat_time_event_df, on=['PATNO'], how='left', \
                                                                          validate='one_to_one')
                grouping_thresh = self.all_feat_thresholds_directions[category][0][grouping][1]
                if not(len(grouping_thresholds_directions) == 1 and grouping_thresholds_directions.keys()[0] == grouping):
                    grouping_feat_time_event_df = self._get_grouping_time_event_df(all_feat_time_event_df, grouping, \
                                                                                   grouping_thresholds_directions.keys(), \
                                                                                   grouping_thresh)
                    all_feat_time_event_df = all_feat_time_event_df.merge(grouping_feat_time_event_df, on=['PATNO'], \
                                                                          how='left', validate='one_to_one')
            if not(len(category_thresholds_directions) == 1 and category_thresholds_directions.keys()[0] == category):
                category_feat_time_event_df \
                    = self._get_grouping_time_event_df(all_feat_time_event_df, category, \
                                                       self.all_feat_thresholds_directions[category][0].keys(), \
                                                       self.all_feat_thresholds_directions[category][1])
                all_feat_time_event_df = all_feat_time_event_df.merge(category_feat_time_event_df, on=['PATNO'], how='left', \
                                                                      validate='one_to_one')
        hybrid_feat_time_event_df = self._get_grouping_time_event_df(all_feat_time_event_df, 'hybrid', \
                                                                     self.all_feat_thresholds_directions.keys(), \
                                                                     self.hybrid_threshold)
        all_feat_time_event_df = all_feat_time_event_df.merge(hybrid_feat_time_event_df, on=['PATNO'], how='left', \
                                                              validate='one_to_one')
        if 'Motor' in self.all_feat_thresholds_directions.keys():
            all_feat_time_event_df['hybrid_requiremotor_E'] = np.where(all_feat_time_event_df['Motor_E'] == 0, 0, \
                                                                       all_feat_time_event_df['hybrid_E'])
            '''
            Motor | Hybrid | Time
            -------------------------
            Obs   | Obs    | Later
            Obs   | Cens   | Hybrid
            Cens  | Obs    | Motor
            Cens  | Cens   | Earlier
            '''
            all_feat_time_event_df['hybrid_requiremotor_T'] \
                = np.where(all_feat_time_event_df['Motor_E'] == 1, \
                           np.where(all_feat_time_event_df['hybrid_E'] == 1, \
                                    all_feat_time_event_df[['Motor_T', 'hybrid_T']].max(axis=1), \
                                    all_feat_time_event_df['hybrid_T']), \
                           np.where(all_feat_time_event_df['hybrid_E'] == 1, \
                                    all_feat_time_event_df['Motor_T'], \
                                    all_feat_time_event_df[['Motor_T', 'hybrid_T']].min(axis=1)))
        return all_feat_time_event_df.reset_index(drop=True)
    
    def plot_hybrid_categories_multi_cohorts(self, df_dict, output_dir):
        '''
        df_dict maps cohort name to cohort dataframe
        cohort dataframes have PATNO, cat1_T, cat1_E, cat2_T, cat2_E, etc., hybrid_T, hybrid_E as columns
        plots Kaplan-Meier curves using lifelines for hybrid outcome and each category's outcome
        saves plots to output_dir + 'hybrid.pdf', 'cat1.pdf', etc.
        '''
        cohorts = df_dict.keys()
        cohorts.sort()
        categories = self.all_feat_thresholds_directions.keys() + ['hybrid']
        if 'Motor' in self.all_feat_thresholds_directions.keys():
            categories.append('hybrid_requiremotor')
        for category in categories:
            first = True
            linestyles = ['dotted', 'dashed', (0, (5, 1)), (0, (3, 5, 1, 5)), 'solid', 'dashdot', (0, (3, 10, 1, 10, 1, 10)), \
                          (0, (3, 1, 1, 1, 1, 1))]
            for cohort_idx in range(len(cohorts)):
                cohort = cohorts[cohort_idx]
                cohort_df = df_dict[cohort]
                assert {category + '_T', category + '_E'}.issubset(set(cohort_df.columns.values.tolist()))
                kmf = KaplanMeierFitter()
                kmf.fit(cohort_df[category + '_T'], event_observed=cohort_df[category + '_E'], label=cohort, linewidth=2)
                if first:
                    ax = kmf.plot(linestyle=linestyles[cohort_idx])
                    first = False
                else:
                    ax = kmf.plot(ax=ax, linestyle=linestyles[cohort_idx], linewidth=2)
                plt.xlabel('Years')
                plt.ylabel('Proportion of population')
                if category == 'hybrid_requiremotor':
                    plt_title = 'hybrid outcome'
                else:
                    plt_title = category + ' outcome'
                plt.title(plt_title)
                plt.savefig(output_dir + category + '.pdf')
                
    def store_order_counts_multi_cohorts(self, df_dict, output_dir):
        '''
        df_dict maps cohort name to cohort dataframe
        cohort dataframes have PATNO, group1_T, group1_E, group2_T, group2_E, etc., 
            cat1_T, cat1_E, cat2_T, cat2_E, etc., hybrid_T, hybrid_E as columns
        stores dictionaries for each category mapping cohort to count dictionary
        stores text files writing top 10 orders for each cohort
        these files are at output_dir + cat1 + '.pkl' or '.txt'
        '''
        cohorts = df_dict.keys()
        cohorts.sort()
        categories = self.all_feat_thresholds_directions.keys() + ['hybrid']
        if 'Motor' in self.all_feat_thresholds_directions.keys():
            categories.append('hybrid_requiremotor')
        for category in categories:
            category_count_dict = dict()
            category_output_str = ''
            for cohort in cohorts:
                cohort_df = df_dict[cohort]
                if category in {'hybrid', 'hybrid_requiremotor'}:
                    feat_list = self.all_feat_thresholds_directions.keys()
                else:
                    feat_list = self.all_feat_thresholds_directions[category][0].keys()
                feat_T_list = [feat + '_T' for feat in feat_list]
                feat_E_list = [feat + '_E' for feat in feat_list]
                assert set(feat_T_list + feat_E_list).issubset(set(cohort_df.columns.values.tolist()))
                order_count_dict, order_count_str = self.get_order_counts(cohort_df, category)
                category_count_dict[cohort] = order_count_dict
                category_output_str += cohort + '\n' + order_count_str + '\n'
            with open(output_dir + category + '.pkl', 'w') as f:
                pickle.dump(category_count_dict, f)
            with open(output_dir + category + '.txt', 'w') as f:
                f.write(category_output_str)
    
    def print_latex_for_dict(self, output_dir):
        '''
        prints the following tables:
        (2-column table)
        \begin{table}
            \centering
            \begin{tabular}{|p{8cm}|p{8cm}|}
                \hline
                \multicolumn{2}{|c|}{$\ge$ hybrid_thresh categories} \\
                \hline
                \ge cat1_thresh cat1 features & $\ge$ cat2_thresh cat2 features \\
                \ge cat2_thresh cat2 features & \\
                \hline
            \end{tabular}
            \caption{\textbf{Hybrid} outcome \textbf{definition} using \textbf{totals/questions}}
            \label{surv_outcome:table:hybrid_totals/questions}
        \end{table}
        (2-column table)
        \begin{table}
            \centering
            \begin{tabular}{|p{8cm}|p{8cm}|}
                \hline
                \multicolumn{2}{|c|}{$\ge$ cat1_thresh cat1 features} \\
                \hline
                feat1 $\ge$ feat1_thresh & feat2 $\le$ feat2_thresh \\
                \ge group1_thresh group1 features & \\
                \hline
            \end{tabular}
            \caption{\textbf{Cat1} outcome \textbf{definition} using \textbf{totals/questions}}
            \label{surv_outcome:table:cat1_totals/questions}
        \end{table}
        (2-column table)
        \begin{table}
            \centering
            \begin{tabular}{|p{8cm}|p{8cm}|}
                \hline
                \multicolumn{2}{|c|}{$\ge$ group1_thresh group1 features} \\
                \hline
                group1_feat1 $\ge$ group1_feat1_thresh & group1_feat2 $\le$ group1_feat2_thresh \\
                \hline
            \end{tabular}
            \caption{\textbf{Cat1 groupings} for outcome \textbf{definition} using \textbf{totals/questions}}
            \label{surv_outcome:table:cat1_totals/questions_groupings}
        \end{table}       
        ... (repeat last 2 for each category)
        feat1, feat2, etc. are replaced by looking up their human readable expressions in human_readable_dict
        writes to a text file called outcome_tables.txt in output_dir
        if feat $\le$ min, print feat $=$ min. if feat $\ge$ max, print feat $=$ min. Same with max # of features in a group/cat. 
        '''
        output_str = ''
        hybrid_table_str = '\\begin{table}\n' + '\t\\centering\n' + '\t\\begin{tabular}{|p{8cm}|p{8cm}|}\n' \
            + '\t\t\\hline\n' + '\t\t\\multicolumn{2}{|c|}{'
        if len(self.all_feat_thresholds_directions) != self.hybrid_threshold:
            hybrid_table_str += ' $\\ge$ '
        hybrid_table_str += str(self.hybrid_threshold) + ' categories} \\\\\n' + '\t\t\\hline\n' + '\t\t'
        hybrid_left_col = True
        for category in self.all_feat_thresholds_directions.keys():
            if self.all_feat_thresholds_directions[category][1] != len(self.all_feat_thresholds_directions[category][0]):
                hybrid_table_str += '$\\ge$ '
            hybrid_table_str += str(self.all_feat_thresholds_directions[category][1]) + ' ' + category + ' features '
            if hybrid_left_col:
                hybrid_table_str += '& '
            else:
                hybrid_table_str += '\\\\\n\t\t\\hline\n\t\t'
            hybrid_left_col = not hybrid_left_col
            category_table_str = '\\begin{table}\n' + '\t\\centering\n' + '\t\\begin{tabular}{|p{8cm}|p{8cm}|}\n' \
                + '\t\t\\hline\n' + '\t\t\\multicolumn{2}{|c|}{'
            if self.all_feat_thresholds_directions[category][1] != len(self.all_feat_thresholds_directions[category][0]):
                category_table_str += '$\\ge$ '
            category_table_str += str(self.all_feat_thresholds_directions[category][1]) + ' ' + category + ' features} ' \
                + ' \\\\\n' + '\t\t\\hline\n' + '\t\t'
            category_grouping_table_str = '\\begin{table}\n' + '\t\\centering\n' + '\t\\begin{tabular}{|p{8cm}|p{8cm}|}\n' \
                + '\t\t\\hline\n'
            left_col = True
            no_groupings = True
            for grouping in self.all_feat_thresholds_directions[category][0]:
                grouping_feats = self.all_feat_thresholds_directions[category][0][grouping][0]
                if len(grouping_feats) == 1:
                    single_feat = grouping_feats.keys()[0]
                    category_table_str += self.human_readable_dict[single_feat].replace('_', '\\_')
                    if grouping_feats[single_feat][1]:
                        if grouping_feats[single_feat][0] == self.min_max_dict[single_feat][1]:
                            category_table_str += ' $=$ '
                        else:
                            category_table_str += ' $\\ge$ '
                    else:
                        if grouping_feats[single_feat][0] == self.min_max_dict[single_feat][0]:
                            category_table_str += ' $=$ '
                        else:
                            category_table_str += ' $\\le$ '
                    category_table_str += str(grouping_feats[single_feat][0])
                else:
                    no_groupings = False
                    if self.all_feat_thresholds_directions[category][0][grouping][1] \
                        != len(self.all_feat_thresholds_directions[category][0][grouping][0]):
                        category_table_str += '$\\ge$ '
                    category_table_str += str(self.all_feat_thresholds_directions[category][0][grouping][1]) \
                        + ' ' + grouping.replace('_', '\\_') + ' features'
                    grouping_left_col = True
                    category_grouping_table_str += '\t\t\\multicolumn{2}{|c|}{'
                    if self.all_feat_thresholds_directions[category][0][grouping][1] \
                        != len(self.all_feat_thresholds_directions[category][0][grouping][0]):
                        category_grouping_table_str += '$\\ge$ '
                    category_grouping_table_str += str(self.all_feat_thresholds_directions[category][0][grouping][1]) + ' ' \
                        + grouping.replace('_', '\\_') + ' features} \\\\\n\t\t\hline\n\t\t'
                    for feat in grouping_feats.keys():
                        category_grouping_table_str += self.human_readable_dict[feat].replace('_', '\\_')
                        if grouping_feats[feat][1]:
                            if grouping_feats[feat][0] == self.min_max_dict[feat][1]:
                                category_grouping_table_str += ' $=$ '
                            else:
                                category_grouping_table_str += ' $\\ge$ '
                        else:
                            if grouping_feats[feat][0] == self.min_max_dict[feat][0]:
                                category_grouping_table_str += ' $=$ '
                            else:
                                category_grouping_table_str += ' $\\le$ '
                        category_grouping_table_str += str(grouping_feats[feat][0])
                        if grouping_left_col:
                            category_grouping_table_str += ' & '
                        else:
                            category_grouping_table_str += ' \\\\\n' + '\t\t\\hline\n' + '\t\t'
                        grouping_left_col = not grouping_left_col
                    if not grouping_left_col:
                        category_grouping_table_str += '\\\\\n' + '\t\t\\hline\n' + '\t\t'
                if left_col:
                        category_table_str += ' & '
                else:
                    category_table_str += ' \\\\\n' + '\t\t\\hline\n' + '\t\t'
                left_col = not left_col
            if not left_col:
                category_table_str += '\\\\\n' + '\t\t\\hline\n' + '\t'
            else:
                category_table_str = category_table_str[:-2] # remove extra \t at end
            category_grouping_table_str = category_grouping_table_str[:-2] # remove extra \t at end
            category_table_str += '\t\\end{tabular}\n' + '\t\\caption{\\textbf{' + category \
                + '} for outcome \\textbf{definition} using \\textbf{' + self.totals_or_questions + '}}\n' \
                + '\t\label{surv_outcome:table:' + category + '_' + self.totals_or_questions + '}\n' \
                + '\\end{table}'
            category_grouping_table_str += '\t\\end{tabular}\n' + '\t\\caption{\\textbf{' + category \
                + ' groupings} for outcome \\textbf{definition} using \\textbf{' + self.totals_or_questions + '}}\n' \
                + '\t\label{surv_outcome:table:' + category + '_' + self.totals_or_questions + '_groupings}\n' \
                + '\\end{table}'
            output_str += category_table_str + '\n\n'
            if not no_groupings:
                output_str += category_grouping_table_str + '\n\n'
        if not hybrid_left_col:
                hybrid_table_str += '\\\\\n' + '\t\t\\hline\n' + '\t'
        else:
            hybrid_table_str = hybrid_table_str[:-2] # remove extra \t at end
        hybrid_table_str += '\\end{tabular}\n' + '\t\\caption{\\textbf{Hybrid} outcome \\textbf{definition} using \\textbf{' \
            + self.totals_or_questions + '}}\n' + '\t\\label{surv_outcome:table:hybrid_' + self.totals_or_questions + '}\n' \
            + '\\end{table}\n\n'
        with open(output_dir + 'outcome_tables.txt', 'w') as f:
            f.write(hybrid_table_str + output_str)
            
    def get_patient_timeline(self, patno_df):
        '''
        patno_df has PATNO, feat1_T, feat1_E, feat2_T, feat2_E, etc., cat1_T, cat1_E, cat2_T, cat2_E, etc.
        hybrid_T, hybrid_E as columns
        prints the following table:
        (2-column table)
        \begin{table}
            \centering
            \begin{tabular}{|c|p{8cm}|p{8cm}|}
                \hline
                \multicolumn{2}{|c|}{Patient observed/censored at xxx} \\
                \hline
                \multicolumn{2}{|c|}{Earliest cat1 observed at xxx} \\
                \hline
                time1 & cat1 feat1 obs at time1 & cat1 feat2 obs at time1 \\
                & cat1 group3 obs at time1 & \\
                \hline
                time2 & cat1 feat1 obs at time2 & \\
                \hline
                ... (Other observed categories)
                ... (Censored categories, ordered by time, + their observed features)
                \hline
            \end{tabular}
            \caption{\textbf{Timeline} for patient \textbf{PATNO} using \textbf{totals/questions}}
            \label{surv_outcome:table:PATNO_timeline_totals/questions}
        \end{table}
        unlike above methods, this one returns a string
        '''
        assert {'PATNO', 'hybrid_T', 'hybrid_E'}.issubset(set(patno_df.columns.values.tolist()))
        output_str = '\\begin{table}\n' + '\t\\centering\n' + '\t\\begin{tabular}{|c|p{8cm}|p{8cm}|}\n' \
            + '\t\t\\hline\n' + '\t\t\\multicolumn{3}{|c|}{Patient '
        if patno_df['hybrid_E'].values[0] == 1:
            output_str += 'observed'
        else:
            output_str += 'censored'
        output_str += ' at ' + str(patno_df['hybrid_T'].values[0]) + '} \\\\\n' + '\t\t\\hline\n'
        cat_obs_times = []
        cat_obs_names = []
        cat_cens_times = []
        cat_cens_names = []
        for category in self.all_feat_thresholds_directions.keys():
            if patno_df[category + '_E'].values[0] == 1:
                cat_obs_times.append(patno_df[category + '_T'].values[0])
                cat_obs_names.append(category)
            else:
                cat_cens_times.append(patno_df[category + '_E'].values[0])
                cat_cens_names.append(category)
        obs_order_idxs = np.argsort(np.array(cat_obs_times)).tolist()
        cens_order_idxs = np.argsort(np.array(cat_cens_times)).tolist()
        cat_order = np.array(cat_obs_names)[obs_order_idxs].tolist() + np.array(cat_cens_names)[cens_order_idxs].tolist()
        for category in cat_order:
            output_str += '\t\t\multicolumn{3}{|c|}{' + category
            if patno_df[category + '_E'].values[0] == 1:
                output_str += ' observed'
            else:
                output_str += ' censored'
            output_str += ' at ' + str(patno_df[category + '_T'].values[0]) + '}\\\\\n' + '\t\t\\hline\n'
            cat_obs_feats = []
            cat_obs_feat_times = []
            for feat in self.all_feat_thresholds_directions[category][0].keys():
                if patno_df[feat + '_E'].values[0] == 1:
                    cat_obs_feats.append(feat)
                    cat_obs_feat_times.append(patno_df[feat + '_T'].values[0])
            if len(cat_obs_feats) == 0:
                output_str += '\t\t\\multicolumn{3}{|c|}{None observed} \\\\\n\t\t\\hline\n'
                continue
            cat_obs_feat_times = np.array(cat_obs_feat_times)
            unique_times = np.unique(cat_obs_feat_times)
            unique_times = np.sort(unique_times)
            for time in unique_times:
                feats_at_time_idxs = np.nonzero(np.where(cat_obs_feat_times == time, 1, 0))[0].tolist()
                output_str += '\t\t\\multirow{' + str(int(math.ceil(len(feats_at_time_idxs)/2.))) + '}{*}{' + str(time) \
                    + '} & '
                time_left_col = True
                for feat in np.array(cat_obs_feats)[feats_at_time_idxs]:
                    if feat in self.human_readable_dict.keys():
                        output_str += self.human_readable_dict[feat]
                    else:
                        output_str += feat
                    if time_left_col:
                        output_str += ' & '
                    else:
                        output_str += ' \\\\\n\t\t\\cline{2-3}\n\t\t& '
                    time_left_col = not time_left_col
                if not time_left_col:
                    output_str += '\\\\\n\t\t\hline\n'
                else:
                    output_str = output_str[:int(-1*len('cline{2-3}\n\t\t& '))] + 'hline\n'
        patno = str(patno_df['PATNO'].values[0])
        output_str += '\t\\end{tabular}\n' + '\t\\caption{\\textbf{Timeline} for patient \\textbf{' + patno \
            + '} using \\textbf{' + self.totals_or_questions + '}. Time is years since enrollment.}\n' \
            + '\t\\label{surv_outcome:table:' + patno + '_timeline_' + self.totals_or_questions + '}\n' \
            + '\\end{table}\n\n'
        return output_str
    
    