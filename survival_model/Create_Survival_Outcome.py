from SurvivalOutcomeBuilder import SurvivalOutcomeBuilder
from SurvivalOutcomeCalculator import SurvivalOutcomeCalculator
import numpy as np, pandas as pd, os, sys, pickle

'''
Take in outcome directory, data directory, and 2 parameters: number of years and proportion of population.
Expecting specification to be in specs.pkl in outcome directory
Outputs will go to a subdirectory in outcome_directory called 'set_' + # years + '_' + prop pop
'''
if len(sys.argv) != 5:
    print('Expecting path to outcome directory, data directory, number of years, and proportion of population as parameters.')
    sys.exit()
outcome_dir = sys.argv[1]
if outcome_dir[-1] != '/':
    outcome_dir += '/'
assert os.path.isdir(outcome_dir)
if 'questions' in outcome_dir:
    totals_or_questions = 'questions'
elif 'subtotals' in outcome_dir:
    totals_or_questions = 'subtotals'
    from Specs_for_subtotals_outcome import compute_subtotals_df
elif 'totals' in outcome_dir:
    totals_or_questions = 'totals'
else:
    print('Whether outcome definition uses totals, subtotals, or questions unclear from outcome directory name.' \
          + 'Please rename directory to include totals, subtotals, or questions.')
    sys.exit()
pipeline_dir = sys.argv[2]
if pipeline_dir[-1] != '/':
    pipeline_dir += '/'
assert os.path.isdir(pipeline_dir)
num_years = float(sys.argv[3])
prop_pop = float(sys.argv[4])
assert prop_pop > 0
assert prop_pop < 1
assert num_years > 0
with open(outcome_dir + 'specs.pkl', 'r') as f:
    spec_dict = pickle.load(f)
all_feats = set([feat for category in spec_dict.keys() for group in spec_dict[category].keys() \
                 for feat in spec_dict[category][group].keys()])
with open(outcome_dir + 'human_readable_dict.pkl', 'r') as f:
    human_readable_dict = pickle.load(f)
with open(outcome_dir + 'min_max_dict.pkl', 'r') as f:
    min_max_dict = pickle.load(f)
outcome_dir += 'set_' + str(num_years) + '_' + str(prop_pop) + '/'

if not os.path.isdir(outcome_dir):
    os.makedirs(outcome_dir)
else: # remove old files
    for f in os.listdir(outcome_dir):
        os.remove(outcome_dir + f)

'''
Read cohort_data_dict. Take maximum of MDS-UPDRS part III questions at same visit, regardless of treatment status
'''
def handle_mdsupdrs3(df):
    for col in df.columns:
        if col.endswith('_untreated'):
            col_root = col[:int(-1*len('_untreated'))]
            col_4variants = [col_root + '_untreated', col_root + '_on', col_root + '_off', col_root + '_unclear']
            assert set(col_4variants).issubset(set(df.columns.values.tolist()))
            df[col_root] = np.nanmax(df[col_4variants], axis=1)
    return df

cohorts = ['PD','SWEDD','HC','GENPD','GENUN','REGPD','REGUN','PRODROMA']
cohort_data_dict = dict()
first = True
for cohort in cohorts:
    cohort_totals_df = pd.read_csv(pipeline_dir + cohort + '_totals_across_time.csv')
    cohort_totals_df = cohort_totals_df.dropna(subset=['PATNO','EVENT_ID_DUR'])
    cohort_totals_df = handle_mdsupdrs3(cohort_totals_df)
    cohort_totals_df['STAI'] = cohort_totals_df[['STATE_ANXIETY','TRAIT_ANXIETY']].sum(axis=1)
    cohort_questions_df = pd.read_csv(pipeline_dir + cohort + '_questions_across_time.csv')
    cohort_questions_df = cohort_questions_df.dropna(subset=['PATNO','EVENT_ID_DUR'])
    if totals_or_questions == 'subtotals':
        cohort_questions_df = compute_subtotals_df(cohort_questions_df)
    else:
        cohort_questions_df = handle_mdsupdrs3(cohort_questions_df)
    cohort_other_df = pd.read_csv(pipeline_dir + cohort + '_other_across_time.csv')
    cohort_other_df = cohort_other_df.dropna(subset=['PATNO','EVENT_ID_DUR'])
    cohort_other_df = handle_mdsupdrs3(cohort_other_df)
    if first:
        totals_cols_used = set(cohort_totals_df.columns.values.tolist()).intersection(all_feats)
        questions_cols_used = set(cohort_questions_df.columns.values.tolist()).intersection(all_feats)
        other_cols_used = set(cohort_other_df.columns.values.tolist()).intersection(all_feats)
        assert len(totals_cols_used.intersection(questions_cols_used)) == 0
        assert len(totals_cols_used.intersection(other_cols_used)) == 0
        assert len(questions_cols_used.intersection(other_cols_used)) == 0
        assert len(all_feats) == len(totals_cols_used) + len(questions_cols_used) + len(other_cols_used)
    else:
        assert totals_cols_used.issubset(set(cohort_totals_df.columns.values.tolist()))
        assert questions_cols_used.issubset(set(cohort_questions_df.columns.values.tolist()))
        assert other_cols_used.issubset(set(cohort_other_df.columns.values.tolist()))
    cohort_totals_df = cohort_totals_df[['PATNO', 'EVENT_ID_DUR'] + list(totals_cols_used)]
    cohort_questions_df = cohort_questions_df[['PATNO', 'EVENT_ID_DUR'] + list(questions_cols_used)]
    cohort_other_df = cohort_other_df[['PATNO', 'EVENT_ID_DUR'] + list(other_cols_used)]
    cohort_df = cohort_totals_df.merge(cohort_questions_df, on=['PATNO', 'EVENT_ID_DUR'], how='outer', validate='one_to_one')
    cohort_data_dict[cohort] = cohort_df.merge(cohort_other_df, on=['PATNO', 'EVENT_ID_DUR'], how='outer', validate='one_to_one')
    
'''
Build survival outcome and store it.
'''
sob = SurvivalOutcomeBuilder(num_years, prop_pop)
sob_output = sob.get_all_thresholds(cohort_data_dict['PD'], spec_dict)
if sob_output is None:
    print('Outcome not possible with these specifications.')
    sys.exit()
outcome_def_dict, hybrid_threshold, pd_time_event_df = sob_output  
with open(outcome_dir + 'outcome_def_'+str(hybrid_threshold)+'.pkl', 'w') as f:
    pickle.dump(outcome_def_dict, f)

'''
Get time of event dataframe for outcome. Plot survival curves. Store order of event counts.
'''
soc = SurvivalOutcomeCalculator(outcome_def_dict, hybrid_threshold, totals_or_questions, human_readable_dict, min_max_dict)
cohort_time_event_dict = dict()
cohort_time_event_dict['PD'] = pd_time_event_df
for cohort in cohort_data_dict.keys():
    if cohort == 'PD':
        continue
    cohort_time_event_dict[cohort] = soc.get_time_event_df(cohort_data_dict[cohort])
with open(outcome_dir + 'cohorts_time_event_dict.pkl', 'w') as f:
    pickle.dump(cohort_time_event_dict, f)
soc.plot_hybrid_categories_multi_cohorts(cohort_time_event_dict, outcome_dir)
soc.store_order_counts_multi_cohorts(cohort_time_event_dict, outcome_dir)
soc.print_latex_for_dict(outcome_dir)

# pick 10 PD patients, 5 prodromal patients, and 5 healthy control patients to make timeline tables for
np.random.seed(18907)
pd_patnos = cohort_time_event_dict['PD'].PATNO.unique()
np.random.shuffle(pd_patnos)
patno_timelines_str = 'PD\n'
for idx in range(10):
    patno = pd_patnos[idx]
    patno_timelines_str += str(patno) + '\n'
    patno_df = cohort_time_event_dict['PD'].loc[cohort_time_event_dict['PD']['PATNO']==patno]
    patno_timelines_str += soc.get_patient_timeline(patno_df)
prodromal_patnos = cohort_time_event_dict['PRODROMA'].PATNO.unique()
np.random.shuffle(prodromal_patnos)
patno_timelines_str += '\nProdromal\n'
for idx in range(5):
    patno = prodromal_patnos[idx]
    patno_df = cohort_time_event_dict['PRODROMA'].loc[cohort_time_event_dict['PRODROMA']['PATNO']==patno]
    patno_timelines_str += soc.get_patient_timeline(patno_df)
hc_patnos = cohort_time_event_dict['HC'].PATNO.unique()
np.random.shuffle(hc_patnos)
patno_timelines_str += '\nHC\n'
for idx in range(5):
    patno = hc_patnos[idx]
    patno_df = cohort_time_event_dict['HC'].loc[cohort_time_event_dict['HC']['PATNO']==patno]
    patno_timelines_str += soc.get_patient_timeline(patno_df)
with open(outcome_dir + 'patient_timeline_tables.txt', 'w') as f:
    f.write(patno_timelines_str)
