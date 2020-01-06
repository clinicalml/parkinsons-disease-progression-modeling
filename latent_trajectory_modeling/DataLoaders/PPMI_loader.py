import numpy as np, pandas as pd
from random import shuffle
import collections
from sklearn.model_selection import KFold

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Synthesize_data import SYN_DATA

class PPMI_loader(object):
    
    def __init__(self, path_to_data, features='MDS-UPDRS II & III untreated'):
        assert features in {'MDS-UPDRS II & III untreated', 'All assessment totals', 'Extended questions'}
        self.features = features
        if self.features == 'MDS-UPDRS II & III untreated':
            pd_questions_df = pd.read_csv(path_to_data)
            nupdrs23_cols = []
            treated_cols = []
            for col in pd_questions_df.columns:
                if col.endswith('_untreated') or col.startswith('NP2'):
                    nupdrs23_cols.append(col)
                if col.endswith('_on') or col.endswith('_off') or col.endswith('_unclear'):
                    treated_cols.append(col)
            assert len(nupdrs23_cols) > 0 and len(treated_cols) > 0 # to check the correct path_to_data was given
            self.observed_column_names = nupdrs23_cols
            untreated_df = pd_questions_df[['PATNO','EVENT_ID_DUR','DIS_DUR_BY_CONSENTDT']+nupdrs23_cols].dropna()
            # drop untreated timepoints that occur after treatment
            treated_df = pd_questions_df[['PATNO','EVENT_ID_DUR']+treated_cols].dropna()
            first_treated_times = treated_df.sort_values(by=['EVENT_ID_DUR']).drop_duplicates(subset=['PATNO'])
            for patno in first_treated_times.PATNO.unique():
                patno_first_treated_time = first_treated_times.loc[first_treated_times['PATNO']==patno].EVENT_ID_DUR.values[0]
                untreated_df = untreated_df.loc[np.logical_or(untreated_df['PATNO'] != patno, \
                                                              untreated_df['EVENT_ID_DUR'] < patno_first_treated_time)]
            self.data = untreated_df
        elif self.features == 'All assessment totals':
            totals = ['NUPDRS2','NUPDRS3_untreated', # motor
                      'MOCA','BJLO','LNS','SEMANTIC_FLUENCY','HVLT_discrim_recog','HVLT_immed_recall','HVLT_retent', # cognitive
                      'SCOPA-AUT', # autonomic
                      'STATE_ANXIETY','TRAIT_ANXIETY','QUIP','GDSSHORT', # psychiatric
                      'EPWORTH','REMSLEEP'] #sleep
            pd_totals_df = pd.read_csv(path_to_data)[['PATNO','EVENT_ID_DUR','DIS_DUR_BY_CONSENTDT']+totals]
            assert set(totals).issubset(set(pd_totals_df.columns.values.tolist()))
            # drop untreated timepoints that occur after treatment
            treated_cols = ['NUPDRS3_on','NUPDRS3_off','NUPDRS3_unclear']
            treated_df = pd_questions_df[['PATNO','EVENT_ID_DUR']+treated_cols].dropna()
            first_treated_times = treated_df.sort_values(by=['EVENT_ID_DUR']).drop_duplicates(subset=['PATNO'])
            for patno in first_treated_times.PATNO.unique():
                patno_first_treated_time = first_treated_times.loc[first_treated_times['PATNO']==patno].EVENT_ID_DUR.values[0]
                pd_totals_df = pd_totals_df.loc[np.logical_or(pd_totals_df['PATNO'] != patno, \
                                                              pd_totals_df['EVENT_ID_DUR'] < patno_first_treated_time)]
            # only use NUPDRS2 where NUPDRS3_untreated is available
            pd_totals_df['NUPDRS2_untreated'] = np.where(pd.isnull(pd_totals_df['NUPDRS3_untreated']), \
                                                         float('NaN'), pd_totals_df['NUPDRS2_untreated'])
            del pd_totals_df['NUPDRS2']
            pd_totals_df = pd_totals_df.dropna(subset=['PATNO','EVENT_ID_DUR','DIS_DUR_BY_CONSENTDT'])
            pd_totals_df = pd_totals_df.dropna(subset=['NUPDRS2_untreated']+totals[1:], how='all')
            self.data = pd_totals_df
            # TODO: may modify to use concomitant medications to get only untreated values for other categories as well
            # however, because other categories could start on treatment, we might end up with no data for some patients...
        else:
            self.data = pd.read_csv(path_to_data) # already pre-formatted
            self.observed_column_names = self.data.columns.values.tolist()
            self.observed_column_names.remove('PATNO')
            self.observed_column_names.remove('EVENT_ID_DUR')
            self.observed_column_names.remove('DIS_DUR_BY_CONSENTDT')
            # min-max normalization
            for col in self.observed_column_names:
                col_min = np.nanmin(self.data[col].values)
                col_max = np.nanmax(self.data[col].values)
                self.data[col] = (self.data[col] - col_min)/float(col_max - col_min)
        
    def get_train_valid_test_split(self, train_valid_test_split=[70,10,20], fold=4):
        '''
        train_valid_test_split is a list of percentages in each category. Must sum to 100.
        Split will be performed patient-wise.
        3 data frames (train, valid, test) returned with columns PATNO, EVENT_ID_DUR, DIS_DUR_BY_CONSENTDT, features
        also returns list of feature names
        {fold*test_split}% to {(fold+1)*test_split}% are the test_indices
        '''
        assert sum(train_valid_test_split) == 100
        assert fold >= 0
        assert (fold+1)*train_valid_test_split[2] <= 100
        np.random.seed(10289)
        patnos = self.data.PATNO.unique()
        np.random.shuffle(patnos)
        test_startpoint = int(fold*train_valid_test_split[2]*len(patnos)/100.)
        test_endpoint = int((fold+1)*train_valid_test_split[2]*len(patnos)/100.)
        test_idxs = range(test_startpoint, test_endpoint)
        train_valid_idxs = list(range(test_startpoint)) + list(range(test_endpoint, len(patnos)))
        train_valid_splitpoint = int(float(train_valid_test_split[0])/sum(train_valid_test_split[:2])*len(train_valid_idxs))
        train_idxs = train_valid_idxs[:train_valid_splitpoint]
        valid_idxs = train_valid_idxs[train_valid_splitpoint:]
        train_patnos = set(patnos[train_idxs].tolist())
        valid_patnos = set(patnos[valid_idxs].tolist())
        test_patnos = set(patnos[test_idxs].tolist())
        return self.data.loc[self.data['PATNO'].isin(train_patnos)], \
            self.data.loc[self.data['PATNO'].isin(valid_patnos)], \
            self.data.loc[self.data['PATNO'].isin(test_patnos)], \
            self.observed_column_names
    
    def split_idx_cv_train_val_test(self, patno_list, fold=5, valid_ratio=0.1):
        """
        Split a list of patno into 'fold' sets of train-valid-test. 
        fold - number of sets of train-valid-test patnos
        valid_ratio - the ratio of validation data our of patno_list
        """

        patno_list = np.array(patno_list)
        np.random.shuffle(patno_list)
        num_patient = len(patno_list)
        num_val = int(valid_ratio*num_patient)

        kf= KFold(n_splits=fold)
        kf.get_n_splits(patno_list)
        all_folds = list()
        for train_val_idx, test_idx in kf.split(patno_list):
            np.random.shuffle(train_val_idx)
            val_idx = train_val_idx[:num_val]
            train_idx = train_val_idx[num_val:]
            idx_dict = {"train": patno_list[train_idx],
                        "valid": patno_list[val_idx],
                        "test": patno_list[test_idx]}
            all_folds.append(idx_dict)

        return all_folds

    def extended_get_train_valid_test_split(self, 
                                            train_valid_test_split=[70,10,20], 
                                            longitudinal=False,
                                            longitudinal_length=1,
                                            aging_format=False,
                                            fill_train_with_sample=None,
                                            fold=None,
                                            valid_ratio=0.1):
        '''
        train_valid_test_split is a list of percentages in each category. Must sum to 100.
        '''

        patnos = self.data.PATNO.unique()
        np.random.shuffle(patnos)

        if fold is None:
            assert sum(train_valid_test_split) == 100
            np.random.seed(10289)
            train_valid_splitpoint = int(train_valid_test_split[0]*len(patnos)/100.)
            valid_test_splitpoint = int((train_valid_test_split[0]+train_valid_test_split[1])*len(patnos)/100.)
            train_patnos = set(patnos[:train_valid_splitpoint].tolist())
            valid_patnos = set(patnos[train_valid_splitpoint:valid_test_splitpoint].tolist())
            test_patnos = set(patnos[valid_test_splitpoint:].tolist())

            return self.get_data_dict(train_patnos, valid_patnos, test_patnos,
                                 longitudinal=longitudinal,
                                 longitudinal_length=longitudinal_length,
                                 aging_format=aging_format,
                                 fill_train_with_sample=fill_train_with_sample)
        else:
            assert(type(fold) == int)
            assert(valid_ratio < (fold-1)/fold)
            all_folds = self.split_idx_cv_train_val_test(patnos, fold=fold, valid_ratio=valid_ratio)
            all_return_dict = list()
            for idx_dict in all_folds:
                return_dict = self.get_data_dict(idx_dict["train"], idx_dict["valid"], idx_dict["test"],
                                            longitudinal=longitudinal,
                                            longitudinal_length=longitudinal_length,
                                            aging_format=aging_format,
                                            fill_train_with_sample=fill_train_with_sample)
                all_return_dict.append(return_dict)

            return all_return_dict


    def get_data_dict(self, train_patnos, valid_patnos, test_patnos,
                      longitudinal=False,
                      longitudinal_length=1,
                      aging_format=False,
                      fill_train_with_sample=None):

        return_dict = dict()
        return_dict["train_df"] = self.data.loc[self.data['PATNO'].isin(train_patnos)]
        return_dict["val_df"] = self.data.loc[self.data['PATNO'].isin(valid_patnos)]
        return_dict["test_df"] = self.data.loc[self.data['PATNO'].isin(test_patnos)]
        return_dict["observed_column_names"] = self.observed_column_names

        if fill_train_with_sample is not None:
            return_dict["orig_train_df"] = return_dict["train_df"].copy()
            return_dict["train_df"] = SYN_DATA().append_sample_data(return_dict["train_df"], 
                                                                    "DIS_DUR_BY_CONSENTDT",
                                                                    num_append_data=fill_train_with_sample)
            if aging_format:
                return_dict["aging_orig_train_df"] = self.modify_df_for_aging_format(return_dict["orig_train_df"])

        if not longitudinal:
            if aging_format:
                for df_name in ["train_df", "val_df", "test_df"]:
                    return_dict["aging_" + df_name] = self.modify_df_for_aging_format(return_dict[df_name])

            return return_dict

        # only if longitudinal
        main_train_df = return_dict["train_df"].copy()

        # only consider row with age > 0
        main_train_df = main_train_df.loc[main_train_df["DIS_DUR_BY_CONSENTDT"] > 0]

        all_PATNO = main_train_df["PATNO"].unique()
        num_patno = len(all_PATNO)
        train_lon_df0 = pd.DataFrame(columns=main_train_df.columns)
        if longitudinal_length == 1:
            train_lon_df1 = pd.DataFrame()
        else:
            assert longitudinal_length > 1
            train_lon_df1 = [pd.DataFrame(columns=main_train_df.columns) for _ in range(longitudinal_length)]

        if longitudinal_length > 1:
            main_train_df = SYN_DATA().append_sample_data(main_train_df, 
                                                        "DIS_DUR_BY_CONSENTDT",
                                                        fill_to=longitudinal_length+1)

        for patno in all_PATNO:
            sub_df = main_train_df.loc[main_train_df["PATNO"] == patno].sort_values(by=["DIS_DUR_BY_CONSENTDT"])
            if longitudinal_length == 1:
                train_lon_df0 = train_lon_df0.append(sub_df[:-1])
                train_lon_df1 = train_lon_df1.append(sub_df[1:])
            else:
                train_lon_df0 = train_lon_df0.append(sub_df.iloc[0])
                for i in range(longitudinal_length):
                    train_lon_df1[i] = train_lon_df1[i].append(sub_df.iloc[i+1])


        train_lon_df0.reset_index(drop=True, inplace=True)
        shuffle_order = np.arange(len(train_lon_df0))
        shuffle(shuffle_order)
        train_lon_df0 = train_lon_df0.iloc[shuffle_order]
        train_lon_df0.reset_index(drop=True, inplace=True)

        if longitudinal_length == 1:
            train_lon_df1.reset_index(drop=True, inplace=True)
            train_lon_df1 = train_lon_df1.iloc[shuffle_order]
            train_lon_df1.reset_index(drop=True, inplace=True)
        else:
            assert longitudinal_length > 1
            assert len(train_lon_df1) == longitudinal_length
            for i in range(longitudinal_length):
                train_lon_df1[i].reset_index(drop=True, inplace=True)
                train_lon_df1[i] = train_lon_df1[i].iloc[shuffle_order]
                train_lon_df1[i].reset_index(drop=True, inplace=True)
  
        return_dict["train_lon_df0"] = train_lon_df0
        return_dict["train_lon_df1"] = train_lon_df1

        if aging_format:
            for df_name in ["train_df", "val_df", "test_df", "train_lon_df0"]:
                return_dict["aging_" + df_name] = self.modify_df_for_aging_format(return_dict[df_name])
            if longitudinal_length == 1:
                return_dict["aging_train_lon_df1"] = self.modify_df_for_aging_format(return_dict["train_lon_df1"])
            else:
                assert longitudinal_length > 1
                return_dict["aging_train_lon_df1"] = \
                    [self.modify_df_for_aging_format(lon_df) for lon_df in return_dict["train_lon_df1"]]

        return return_dict

    def modify_df_for_aging_format(self, df):
        df = df.rename(columns={"PATNO":"individual_id", "DIS_DUR_BY_CONSENTDT":"age_sex___age"}).copy()
        df = df.drop(["EVENT_ID_DUR"], axis=1)
        df.loc[:,"individual_id"] = df.index
        df = df.loc[df["age_sex___age"] > 0]
        return df
    
    def get_baseline_data_split(self, path_to_baseline_data, train_valid_test_patnos):
        '''
        splits data into 3 dataframes for train, valid, test
        train_valid_test_patnos: dictionary mapping 'train', 'valid', and 'test' each to a set of PATNOs
        '''
        assert {'train','valid','test'}.issubset(set(train_valid_test_patnos.keys()))
        data = pd.read_csv(path_to_baseline_data)
        assert 'PATNO' in set(data.columns.values.tolist())
        baseline_cols = data.columns.values.tolist()
        baseline_cols.remove('PATNO')
        return data.loc[data['PATNO'].isin(train_valid_test_patnos['train'])], \
               data.loc[data['PATNO'].isin(train_valid_test_patnos['valid'])], \
               data.loc[data['PATNO'].isin(train_valid_test_patnos['test'])], \
               baseline_cols
        
    def get_baseline_human_readable_dict(self):
        '''
        returns a dictionary mapping baseline feature abbreviation to human readable version of it
        '''
        return {'MALE': 'Male', 'RAWHITE': 'White', 'FAMHIST': 'Family history', \
                'EDUCYRS': '# years education', 'RIGHT_HANDED': 'Right-handed', 'UPSIT': 'Smell', \
                'DIS_DUR_BY_CONSENTDT': 'Disease duration', 'Genetic PCA component 0': 'Genetic PCA comp 0', \
                'Genetic PCA component 1': 'Genetic PCA comp 1', 'Genetic PCA component 2': 'Genetic PCA comp 2', \
                'Genetic PCA component 3': 'Genetic PCA comp 3', 'Genetic PCA component 4': 'Genetic PCA comp 4', \
                'Genetic PCA component 5': 'Genetic PCA comp 5', 'Genetic PCA component 6': 'Genetic PCA comp 6', \
                'Genetic PCA component 7': 'Genetic PCA comp 7', 'Genetic PCA component 8': 'Genetic PCA comp 8', \
                'Genetic PCA component 9': 'Genetic PCA comp 9', 'AGE': 'Age', 'RIGHT_DOMSIDE': 'Right-dominant', \
                'SYSSTND': 'Systolic BP standing', 'SYSSUP': 'Systolic BP supine', \
                'HRSTND': 'Heart rate standing', 'HRSUP': 'Heart rate supine', \
                'DIASUP': 'Diastolic BP standing', 'DIASTND': 'Diastolic BP supine', 'TEMPC': 'Temperature (C)', \
                'ipsilateral_putamen': 'DaTscan ipsilateral putamen', 'ipsilateral_caudate': 'DaTscan ipsilateral caudate', \
                'count_density_ratio_ipsilateral': 'DaTscan ipsilateral CDR', \
                'count_density_ratio_contralateral': 'DaTscan contralateral CDR', \
                'contralateral_putamen': 'DaTscan contralateral putamen', \
                'contralateral_caudate': 'DaTscan contralateral caudate',
                'asymmetry_index_caudate': 'DaTscan caudate asymmetry', 'asymmetry_index_putamen': 'DaTscan putamen asymmetry', \
                'WGTKG': 'Weight (kg)', 'HTCM': 'Height (cm)', 'DVT_SDM': 'Symbol digits modality', \
                'PTAU_ABETA_ratio': 'CSF pTau to Abeta', 'TTAU_ABETA_ratio': 'CSF tTau to Abeta', \
                'PTAU_TTAU_ratio': 'CSF pTau to tTau', 'PTAU_log': 'CSF pTau log', 'ABETA_log': 'CSF Abeta log', \
                'ASYNU_log': 'CSF alpha-synuclein log', 'CSF Hemoglobin': 'CSF hemoglobin', \
                'SCOPA-AUT': 'SCOPA-AUT', 'HVLT_discrim_recog': 'HVLT discrim recog', 'STAI': 'STAI (anxiety)', \
                'HVLT_immed_recall': 'HVLT immed recall', 'QUIP': 'QUIP (impulsive)', 'EPWORTH': 'Epworth sleep', \
                'GDSSHORT': 'GDS depressed', 'HVLT_retent': 'HVLT retention', 'BJLO': 'Benton judg line orient', \
                'LNS': 'Letter number seq', 'SEMANTIC_FLUENCY': 'Semantic fluency', 'REMSLEEP': 'RBD sleep disorder', \
                'NUPDRS1': 'MDS-UPDRS I', 'NUPDRS2': 'MDS-UPDRS II', 'NUPDRS3_untreated': 'MDS-UPDRS III', \
                'MOCA': 'Montreal cognitive', \
                'TD_PIGD_untreated:tremor': 'Tremor-dominant', 'TD_PIGD_untreated:posture': 'Postural instability-dominant'}

    def get_human_readable_dict(self):
        '''
        returns a dictionary mapping question abbreviation to human readable version of it
        '''
        if self.features == 'MDS-UPDRS II & III untreated':
            return {'NP2DRES': 'dressing', 'NP2EAT': 'eating', \
                    'NP2HYGN': 'hygiene', 'NP2WALK': 'walking', \
                    'NP2HOBB': 'hobbies', \
                    'NP2RISE': 'getting up 2', \
                    'NP2TURN': 'turning in bed', 'NP2SALV': 'drooling', \
                    'NP2FREZ': 'freezing 2', 'NP2HWRT': 'writing', \
                    'NP2SPCH': 'speech 2', 'NP2TRMR': 'tremor', \
                    'NP2SWAL': 'swallowing', \

                    'NP3RIGLU_untreated': 'rigid UL', \
                    'NP3FACXP_untreated': 'face expr', \
                    'NP3RIGRL_untreated': 'rigid RL', \
                    'NP3KTRML_untreated': 'kinetic trem LH', \
                    'NP3FTAPR_untreated': 'finger tap R', \
                    'NP3RTCON_untreated': 'rest trem constancy', \
                    'NP3PRSPL_untreated': 'pronation-supination LH', \
                    'NP3SPCH_untreated': 'speech 3', \
                    'NP3RTALJ_untreated': 'rest trem amp jaw', \
                    'NP3LGAGR_untreated': 'leg agility R', \
                    'NP3HMOVR_untreated': 'movements RH', \
                    'NP3RISNG_untreated': 'getting up 3', \
                    'NP3RTALU_untreated': 'rest trem amp UL', \
                    'NP3FTAPL_untreated': 'finger tap L', \
                    'NP3RTALL_untreated': 'rest trem amp LL', \
                    'NP3RIGRU_untreated': 'rigid UR', \
                    'NP3TTAPL_untreated': 'toe tap L', \
                    'NP3PSTBL_untreated': 'postural stability', \
                    'NP3RTARL_untreated': 'rest trem amp LR', \
                    'NP3BRADY_untreated': 'bradykinesia', \
                    'NP3RTARU_untreated': 'rest tremor amp UR', \
                    'NP3PRSPR_untreated': 'pronation-supination RH', \
                    'NP3RIGN_untreated': 'rigid neck', \
                    'NP3RIGLL_untreated': 'rigid LL', \
                    'NP3PTRML_untreated': 'postural trem LH', \
                    'NP3TTAPR_untreated': 'toe tap R', \
                    'NP3LGAGL_untreated': 'leg agility L', \
                    'NP3HMOVL_untreated': 'movements LH', 'NP3GAIT_untreated': 'gait', \
                    'NP3PTRMR_untreated': 'postural trem RH', \
                    'NP3KTRMR_untreated': 'kinetic trem RH', \
                    'NP3POSTR_untreated': 'posture', 'NP3FRZGT_untreated': 'freezing 3'}
            '''
            return {'NP2DRES': 'MDS-UPDRS 2.5 dressing', 'NP2EAT': 'MDS-UPDRS 2.4 eating tasks', \
                    'NP2HYGN': 'MDS-UPDRS 2.6 hygiene', 'NP2WALK': 'MDS-UPDRS 2.12 walking and balance', \
                    'NP2HOBB': 'MDS-UPDRS 2.8 hobbies and other activities', \
                    'NP2RISE': 'MDS-UPDRS  2.11 getting out of bed/car/deep chair', \
                    'NP2TURN': 'MDS-UPDRS 2.9 turning in bed', 'NP2SALV': 'MDS-UPDRS 2.2 saliva and drooling', \
                    'NP2FREZ': 'MDS-UPDRS 2.13 freezing', 'NP2HWRT': 'MDS-UPDRS 2.7 handwriting', \
                    'NP2SPCH': 'MDS-UPDRS 2.1 speech', 'NP2TRMR': 'MDS-UPDRS 2.10 tremor', \
                    'NP2SWAL': 'MDS-UPDRS 2.3 chewing and swallowing',\

                    'NP3RIGLU_untreated': 'MDS-UPDRS 3.3 rigidity upper left', \
                    'NP3FACXP_untreated': 'MDS-UPDRS 3.2 facial expression', \
                    'NP3RIGRL_untreated': 'MDS-UPDRS 3.3 rigidity right lower', \
                    'NP3KTRML_untreated': 'MDS-UPDRS 3.16 kinetic tremor of hands left', \
                    'NP3FTAPR_untreated': 'MDS-UPDRS 3.4 finger tapping right', \
                    'NP3RTCON_untreated': 'MDS-UPDRS 3.18 constancy of rest tremor', \
                    'NP3PRSPL_untreated': 'MDS-UPDRS 3.6 pronation-supination movements of hands left', \
                    'NP3SPCH_untreated': 'MDS-UPDRS 3.1 speech', \
                    'NP3RTALJ_untreated': 'MDS-UPDRS 3.17 rest tremor amplitude lip/jaw', \
                    'NP3LGAGR_untreated': 'MDS-UPDRS 3.8 leg agility right', \
                    'NP3HMOVR_untreated': 'MDS-UPDRS 3.5 hand movements right', \
                    'NP3RISNG_untreated': 'MDS-UPDRS 3.9 arising from chair', \
                    'NP3RTALU_untreated': 'MDS-UPDRS 3.17 rest tremor amplitude left upper', \
                    'NP3FTAPL_untreated': 'MDS-UPDRS 3.4 finger tapping left', \
                    'NP3RTALL_untreated': 'MDS-UPDRS 3.17 rest tremor amplitude left lower', \
                    'NP3RIGRU_untreated': 'MDS-UPDRS 3.3 rigidity right upper', \
                    'NP3TTAPL_untreated': 'MDS-UPDRS 3.7 toe tapping left', \
                    'NP3PSTBL_untreated': 'MDS-UPDRS 3.12 postural stability', \
                    'NP3RTARL_untreated': 'MDS-UPDRS 3.17 rest tremor amplitude right lower', \
                    'NP3BRADY_untreated': 'MDS-UPDRS 3.14 global spontaneity of movement/body bradykinesia', \
                    'NP3RTARU_untreated': 'MDS-UPDRS 3.17 rest tremor amplitude right upper', \
                    'NP3PRSPR_untreated': 'MDS-UPDRS 3.6 pronation-supination movements of hands right', \
                    'NP3RIGN_untreated': 'MDS-UPDRS 3.3 rigidity neck', \
                    'NP3RIGLL_untreated': 'MDS-UPDRS 3.3 rigidity lower left', \
                    'NP3PTRML_untreated': 'MDS-UPDRS 3.15 postural tremor of hands left', \
                    'NP3TTAPR_untreated': 'MDS-UPDRS 3.7 toe tapping right', \
                    'NP3LGAGL_untreated': 'MDS-UPDRS 3.8 leg agility left', \
                    'NP3HMOVL_untreated': 'MDS-UPDRS 3.5 hand movements left', 'NP3GAIT_untreated': 'MDS-UPDRS 3.10 gait', \
                    'NP3PTRMR_untreated': 'MDS-UPDRS 3.15 postural tremor of hands right', \
                    'NP3KTRMR_untreated': 'MDS-UPDRS 3.16 kinetic tremor of hands right', \
                    'NP3POSTR_untreated': 'MDS-UPDRS 3.13 posture', 'NP3FRZGT_untreated': 'MDS-UPDRS 3.11 freezing of gait'}
            '''
        elif self.features == 'All assessment totals':
            return {'NUPDRS2_untreated': 'MDS-UPDRS II', 'NUPDRS3_untreated': 'MDS-UPDRS III', 'MOCA': 'MoCA', 'BJLO': 'BJLO', \
                    'LNS': 'LNS', 'SEMANTIC_FLUENCY': 'Semantic fluency', 'HVLT_immed_recall': 'HVLT immed recall', \
                    'HVLT_discrim_recog': 'HVLT discrim recog', 'HVLT_retent': 'HVLT retent', 'GDSSHORT': 'GDS depression', \
                    'QUIP': 'QUIP (impulsive)', 'STAI': 'STAI anxiety', 'SCOPA-AUT': 'SCOPA-AUT', 'EPWORTH': 'EPWORTH', \
                    'REMSLEEP': 'RBD sleep'}
        else:
            raise NotImplementedError
    
    def drop_many_nan_feat(self,dataFrame,col_list,threshold):
        '''
        Find the # of missing observations for each col in col_list
        Remove col from col_list with # of missing obs > threshold
        Return dataFrame with the remaining columns
        '''
        col_set = {'PATNO'} | set(col_list)
        assert col_set.issubset(set(dataFrame.columns.values.tolist()))
        col_nan = dataFrame[col_list].isnull().sum(axis=0).tolist() # cound #of missing obs for each col
        counter=collections.Counter(col_nan)
        print(counter) #columns with same #of missing obs
        col_few_nan = [i for i in range(len(col_nan)) if col_nan[i] < threshold]
        col_list_new = [col_list[idx] for idx in col_few_nan]
        df = dataFrame[['PATNO']+col_list_new]
        return df