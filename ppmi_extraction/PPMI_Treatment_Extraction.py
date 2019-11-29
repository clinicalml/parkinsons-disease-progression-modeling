import pandas as pd, numpy as np, sys, os

if len(sys.argv) != 3:
    print('expecting path to ppmi directory and download date in format YYYYMmmDD as 1st and 2nd parameters')
    sys.exit()

ppmi_dir = sys.argv[1]
download_date = sys.argv[2]
if not ppmi_dir.endswith('/'):
    ppmi_dir += '/'
raw_datadir = ppmi_dir + 'raw_data_asof_' + download_date + '/'
if not os.path.isdir(raw_datadir):
    print(raw_datadir + ' as specified by input parameters does not exist')
    sys.exit()
conco_meds_path = raw_datadir + 'Concomitant_Medications.csv'
conco_meds_df = pd.read_csv(conco_meds_path)

druglist_path = 'DrugList_from_MonicaJ.csv'
druglist_df = pd.read_csv(druglist_path)

modified_druglist_df = druglist_df[['DRUG','GENERIC','CLASS']]
modified_druglist_df['CLASS'] = np.where(modified_druglist_df['CLASS']=='.', 'OTHER', modified_druglist_df['CLASS'])
# assign each generic name to its most common drug class
for drug in set(modified_druglist_df.GENERIC.unique()):
    specific_drug_df = modified_druglist_df.loc[modified_druglist_df['GENERIC']==drug]
    if specific_drug_df.CLASS.nunique() >= 2:
        modified_druglist_df['CLASS'] = np.where(modified_druglist_df['GENERIC']==drug, \
                                                 specific_drug_df.CLASS.value_counts().keys()[0], \
                                                 modified_druglist_df['CLASS'])
# condense some classes
modified_druglist_df['CLASS'] = np.where(modified_druglist_df['CLASS']=='ANTIHYERTENSIVE', 'ANTIHYPERTENSIVE', \
                                         modified_druglist_df['CLASS'])
modified_druglist_df['CLASS'] = np.where(modified_druglist_df['CLASS']=='THYROID?', 'THYROID', \
                                         modified_druglist_df['CLASS'])
modified_druglist_df['CLASS'] = np.where(modified_druglist_df['CLASS'].str.contains('DIGESTIVE AID'), \
                                         'DIGESTIVE AID', modified_druglist_df['CLASS'])
modified_druglist_df['CLASS'] = np.where(modified_druglist_df['CLASS']=='ANTHISTAMINE', 'ANTIHISTAMINE', \
                                         modified_druglist_df['CLASS'])
modified_druglist_df['CLASS'] = np.where(modified_druglist_df['CLASS']=='ANALGESIC BARBITURATE STIMULANT', \
                                         'ANALGESIC', modified_druglist_df['CLASS'])
modified_druglist_df['CLASS'] = np.where(modified_druglist_df['CLASS']=='NSAID ANALGESIC STIMULANT', \
                                         'NSAID', modified_druglist_df['CLASS'])
modified_druglist_df['CLASS'] = np.where(modified_druglist_df['CLASS']=='MUSCLE RELAXANT', \
                                          'MUSCLE RELAXER', modified_druglist_df['CLASS'])
modified_druglist_df['CLASS'] = np.where(modified_druglist_df['CLASS']=='ANTIMIGRAINE', \
                                         'ANALGESIC', modified_druglist_df['CLASS'])
modified_druglist_df['CLASS'] = np.where(modified_druglist_df['CLASS']=='VASODILATOR: ANTIANGINA', \
                                         'VASODILATOR', modified_druglist_df['CLASS'])
modified_druglist_df['CLASS'] = np.where(modified_druglist_df['CLASS']=='ANTIMALARIAL', \
                                         np.where(modified_druglist_df['DRUG']=='MALARONE', \
                                                  'ANTIFUNGAL', 'IMMUNOSUPPRESSANT'), \
                                         modified_druglist_df['CLASS'])

#match WHODRUG and DRUG
conco_meds_df = conco_meds_df.merge(modified_druglist_df[['DRUG','GENERIC','CLASS']].drop_duplicates(), \
                                    left_on=['WHODRUG'], right_on=['DRUG'], how = 'left', \
                                    validate='many_to_one')
# match CMTRT and DRUG
conco_meds_classnan_df = conco_meds_df.loc[pd.isnull(conco_meds_df['CLASS'])]
del conco_meds_classnan_df['GENERIC']
del conco_meds_classnan_df['CLASS']
conco_meds_classnan_df \
    = conco_meds_classnan_df.merge(modified_druglist_df[['DRUG','GENERIC','CLASS']].drop_duplicates(), \
                                   left_on=['CMTRT'], right_on=['DRUG'], how = 'left', \
                                   validate='many_to_one')
conco_meds_classnonnan_df = conco_meds_df.loc[~pd.isnull(conco_meds_df['CLASS'])]
conco_meds_df = pd.concat([conco_meds_classnonnan_df, conco_meds_classnan_df])
drug_x_df = conco_meds_df['DRUG_x']
drug_x_df.rename(columns={'DRUG_x':'DRUG'}, inplace=True)
drug_y_df = conco_meds_df['DRUG_y']
drug_y_df.rename(columns={'DRUG_y':'DRUG'}, inplace=True)
del conco_meds_df['DRUG_x']
del conco_meds_df['DRUG_y']
conco_meds_df.update(drug_x_df, overwrite=False)
conco_meds_df.update(drug_y_df, overwrite=False)
# match DRUG and GENERIC
conco_meds_classnan_df = conco_meds_df.loc[pd.isnull(conco_meds_df['CLASS'])]
del conco_meds_classnan_df['GENERIC']
del conco_meds_classnan_df['CLASS']
generic_modified_druglist_df = modified_druglist_df[['GENERIC','CLASS']].drop_duplicates()
generic_modified_druglist_df['DRUG'] = generic_modified_druglist_df['GENERIC']
conco_meds_classnan_df = conco_meds_classnan_df.merge(generic_modified_druglist_df, on=['DRUG'], how='left', \
                                                      validate='many_to_one')
conco_meds_classnonnan_df = conco_meds_df.loc[~pd.isnull(conco_meds_df['CLASS'])]
conco_meds_df = pd.concat([conco_meds_classnonnan_df, conco_meds_classnan_df])
conco_meds_df['GENERIC'] = np.where(pd.isnull(conco_meds_df['GENERIC']), conco_meds_df['DRUG'], \
                                              conco_meds_df['GENERIC'])
# split ANALGESIC ANTIHISTAMINE into its 2 component classes
analgesic_antihistamine_df = conco_meds_df.loc[conco_meds_df['CLASS']=='ANALGESIC ANTIHISTAMINE']
analgesic_antihistamine_df['CLASS'] = 'ANTIHISTAMINE'
conco_meds_df['CLASS'] = np.where(conco_meds_df['CLASS']=='ANALGESIC ANTIHISTAMINE', \
                                  'ANALGESIC', conco_meds_df['CLASS'])
conco_meds_df = pd.concat([conco_meds_df, analgesic_antihistamine_df])
# group uncommon drug classes into 'RARE_CLASS'
class_value_counts = conco_meds_df.CLASS.value_counts()
rare_class_set = set()
for drug_class in class_value_counts.keys():
    if class_value_counts[drug_class] < 10:
        rare_class_set.add(drug_class)
conco_meds_df['CLASS'] = np.where(conco_meds_df['CLASS'].isin(rare_class_set), \
                                  'RARE_CLASS', conco_meds_df['CLASS'])

# fix start and stop dates where nan
conco_meds_df = conco_meds_df.dropna(subset=['STARTDT','STOPDT','ONGOING'], how='all')
conco_meds_df['STOPDT'] = np.where(np.logical_and(pd.isnull(conco_meds_df['STARTDT']), \
                                                  pd.isnull(conco_meds_df['STOPDT'])), conco_meds_df['ORIG_ENTRY'], \
                                   conco_meds_df['STOPDT'])
conco_meds_df['STARTDT'] = np.where(pd.isnull(conco_meds_df['STARTDT']), conco_meds_df['STOPDT'], conco_meds_df['STARTDT'])
conco_meds_df['STARTDT'] = np.where(conco_meds_df['STARTDT'].str.contains('1015'), \
                                    conco_meds_df['STARTDT'].str.replace('1015','2015'), conco_meds_df['STARTDT'])
conco_meds_df['STARTDT'] = np.where(conco_meds_df['STARTDT'].str.contains('216'), \
                                    conco_meds_df['STARTDT'].str.replace('216','2016'), conco_meds_df['STARTDT'])
conco_meds_df['STARTDT'] = np.where(conco_meds_df['STARTDT'].str.contains('8201'), \
                                    conco_meds_df['STARTDT'].str.replace('8201','2018'), conco_meds_df['STARTDT'])
conco_meds_df['STARTDT'] = pd.to_datetime(conco_meds_df['STARTDT'])
conco_meds_df['STOPDT'] = pd.to_datetime(conco_meds_df['STOPDT'])
drug_classes = list(set(conco_meds_df.CLASS.unique()))

def output_treatment_freq(prior_df, df):
    output_str = 'drug: num_patnos_before_first_visit, num_patnos_current, avg_num_visits_on, avg_num_visits_off\n'
    agg_stats = []
    agg_stat_names = []
    for drug_class in drug_classes:
        num_patnos_prior = prior_df.loc[prior_df[drug_class]==1].PATNO.nunique()
        patnos_used_class = set(df.loc[df[drug_class]==1].PATNO.unique())
        num_patnos = len(patnos_used_class)
        if num_patnos == 0:
            avg_num_visits_on = 0
            avg_num_visits_off = 0
        else:
            patnos_used_meds_df = df.loc[df['PATNO'].isin(patnos_used_class)]
            avg_num_visits_on = len(patnos_used_meds_df.loc[patnos_used_meds_df[drug_class]==1])/float(num_patnos)
            avg_num_visits_off = len(patnos_used_meds_df.loc[patnos_used_meds_df[drug_class]==0])/float(num_patnos)
        agg_stats += [num_patnos_prior, num_patnos, avg_num_visits_on, avg_num_visits_off]
        agg_stat_names += [drug_class + '_num_patnos_before_first_visit', drug_class + '_num_patnos', drug_class + '_avg_num_visits_on', drug_class + '_avg_num_visits_off']
        output_str += drug_class + ': ' + str(num_patnos_prior) + ', ' + str(num_patnos) + ', ' + str(avg_num_visits_on) + ', ' \
            + str(avg_num_visits_off) + '\n'
    return output_str, agg_stats, agg_stat_names

# read files to get visit dates
pipeline_dir = ppmi_dir + 'pipeline_output_asof_' + download_date + '/'
treatment_pipeline_dir = ppmi_dir + 'treatment_pipeline_output_asof_' + download_date + '/'
if not os.path.isdir(treatment_pipeline_dir):
    os.makedirs(treatment_pipeline_dir)
agg_stat_df = pd.read_csv(pipeline_dir + 'agg_stat.csv')
cohorts = agg_stat_df.columns[1:]
files_in_each_cohort = ['_screening.csv', '_baseline.csv', '_totals_across_time.csv', '_questions_across_time.csv', '_other_across_time.csv']
first = True
for cohort in cohorts:
    cohort_df = pd.read_csv(pipeline_dir + cohort + files_in_each_cohort[0])[['PATNO','EVENT_ID','INFODT']].dropna()
    for file in files_in_each_cohort[1:]:
        file_df = pd.read_csv(pipeline_dir + cohort + file)[['PATNO','EVENT_ID','INFODT']].dropna()
        cohort_df = pd.concat([cohort_df, file_df])
    cohort_df['INFODT'] = pd.to_datetime(cohort_df['INFODT'])
    cohort_df = cohort_df.sort_values(by=['PATNO','INFODT'])
    cohort_df = cohort_df.drop_duplicates(subset=['PATNO','EVENT_ID'], keep='first')
    for drug_class in drug_classes:
        cohort_df[drug_class] = 0
    cohort_prior_first_visit_df = cohort_df.drop_duplicates(subset=['PATNO'], keep='first')
    prior_first_patnos = cohort_prior_first_visit_df.PATNO.values.tolist()
    cohort_cols = cohort_df.columns.values.tolist()
    patnos = cohort_df.PATNO.values.tolist()
    infodts = pd.to_datetime(cohort_df.INFODT).values
    prev_patno = -1
    for row_idx in range(len(cohort_df)):
        patno = patnos[row_idx]
        infodt = infodts[row_idx]
        if prev_patno != patno:
            prev_patno = patno
            prev_infodt = pd.to_datetime(infodt - pd.Timedelta(3, units='M')) # for first visit, current treatment is everything within 3 months prior; everything before that is treatment history
            patno_meds_before_enroll_df = conco_meds_df.loc[np.logical_and(conco_meds_df['PATNO']==patno, \
                                                                           conco_meds_df['STARTDT'] < prev_infodt)]
            prior_row_idx = prior_first_patnos.index(patno)
            for drug_class in set(patno_meds_before_enroll_df.CLASS.unique()):
                cohort_prior_first_visit_df.iloc[prior_row_idx, cohort_cols.index(drug_class)] = 1
        # everything between previous visit and current visit is considered current treatment
        patno_meds_on_infodt_df = conco_meds_df.loc[np.logical_and.reduce((conco_meds_df['PATNO']==patno, \
                                                                           conco_meds_df['STARTDT'] < infodt, \
                                                                           np.logical_or(conco_meds_df['STOPDT'] >= prev_infodt, \
                                                                                         pd.isnull(conco_meds_df['STOPDT']))))]
        for drug_class in set(patno_meds_on_infodt_df.CLASS.unique()):
            cohort_df.iloc[row_idx, cohort_cols.index(drug_class)] = 1
        prev_infodt = infodt
    
    # output csvs and aggregate stats
    cohort_prior_first_visit_df.to_csv(treatment_pipeline_dir + cohort + '_treatment_before_first_visit.csv', index=False)
    cohort_df.to_csv(treatment_pipeline_dir + cohort + '_treatment_between_visits.csv', index=False)
    output_str, agg_stats, agg_stat_names = output_treatment_freq(cohort_prior_first_visit_df, cohort_df)
    with open(treatment_pipeline_dir + cohort + '_summary.txt', 'w') as f:
        f.write(cohort + '\n' + output_str)
    if first:
        agg_stat_df = pd.DataFrame(agg_stat_names, columns=['agg_stats'])
        first = False
    agg_stat_df[cohort] = agg_stats
agg_stat_df.to_csv(treatment_pipeline_dir + 'agg_stats.csv', index=False)