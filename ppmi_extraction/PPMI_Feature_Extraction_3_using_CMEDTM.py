import pandas as pd, numpy as np, pickle, sys, os
from scipy.stats.mstats import gmean

if len(sys.argv) != 3:
    print('expecting path to ppmi directory and download date in format YYYYMmmDD as 1st and 2nd parameters')
    sys.exit()

ppmi_dir = sys.argv[1]
download_date = sys.argv[2]
if not ppmi_dir[-1].endswith('/'):
    ppmi_dir += '/'
raw_datadir = ppmi_dir + 'raw_data_asof_' + download_date + '/'
if not os.path.isdir(raw_datadir):
    print(raw_datadir + ' as specified by input parameters does not exist')
    sys.exit()

symptom_cols = []
symptom_cols_dict = dict()
total_cols = []
standard3_cols = ['PATNO','EVENT_ID', 'INFODT']
shared_cols = []
shared_cols_dict = dict()
screening_cols = []
screening_cols_dict = dict()
baseline_cols = []
baseline_cols_dict = dict()
infodt_only_cols = [] # to avoid duplicates in count availability later
eventid_only_cols = []

def average_by_patno_eventid(df):
    df_standard3_cols = df[standard3_cols].drop_duplicates(subset=['PATNO','EVENT_ID'])
    del df['INFODT']
    agg_dict = dict()
    for col in df.columns:
        if col == 'PATNO' or col == 'EVENT_ID':
            continue
        col_agg_dict = dict()
        col_agg_dict[col] = np.nanmean
        agg_dict[col] = col_agg_dict
    mean_df = df.groupby(by=['PATNO','EVENT_ID']).agg(agg_dict)
    mean_df.columns = mean_df.columns.droplevel(0)
    mean_df = mean_df.reset_index()
    assert len(mean_df) == len(df_standard3_cols)
    mean_df = mean_df.merge(df_standard3_cols, on=['PATNO','EVENT_ID'], validate='one_to_one')
    return mean_df

def merge_dfs(collection_df, specific_df):
    # if needed, first take average if any 2 rows in specific_df share the same PATNO and EVENT_ID (keep either of the INFODT)
    # INFODT column should use collection_df's first, then specific_df's
    if len(specific_df) != len(specific_df.drop_duplicates(subset=['PATNO','EVENT_ID'])):
        specific_df = average_by_patno_eventid(specific_df)
    collection_df = collection_df.merge(specific_df, on=['PATNO','EVENT_ID'], how='outer', copy=False, validate = 'one_to_one')
    infodt_y_df = collection_df[['INFODT_y']]
    infodt_y_df.rename(columns={'INFODT_y':'INFODT'}, inplace=True)
    del collection_df['INFODT_y']
    collection_df.rename(columns={'INFODT_x':'INFODT'}, inplace=True)
    infodt_y_df['INFODT'] = pd.to_datetime(infodt_y_df['INFODT'])
    collection_df['INFODT'] = pd.to_datetime(collection_df['INFODT'])
    collection_df.update(infodt_y_df, overwrite=False)
    return collection_df

# MDS-UPDRS III
raw_mds_updrs3_path = raw_datadir + 'MDS_UPDRS_Part_III.csv'
raw_mds_updrs3_df = pd.read_csv(raw_mds_updrs3_path)
raw_mds_updrs3_df['LAST_UPDATE'] = pd.to_datetime(raw_mds_updrs3_df['LAST_UPDATE'])
raw_mds_updrs3_df['INFODT'] = pd.to_datetime(raw_mds_updrs3_df['INFODT'])
raw_mds_updrs3_df = raw_mds_updrs3_df.sort_values(by=standard3_cols + ['LAST_UPDATE'])
raw_mds_updrs3_df = raw_mds_updrs3_df.drop_duplicates(subset={'PATNO','EVENT_ID','INFODT','PD_MED_USE','CMEDTM'}, keep='last')
raw_mds_updrs3_df.rename(columns={'PN3RIGRL': 'NP3RIGRL'}, inplace=True)
updrs3_symptom_cols = ['NP3SPCH', 'NP3FACXP', 'NP3RIGN', 'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL', 'NP3FTAPR', \
                       'NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL', 'NP3PRSPR', 'NP3PRSPL', 'NP3TTAPR', 'NP3TTAPL', 'NP3LGAGR', \
                       'NP3LGAGL', 'NP3RISNG', 'NP3GAIT', 'NP3FRZGT', 'NP3PSTBL', 'NP3POSTR', 'NP3BRADY', 'NP3PTRMR', \
                       'NP3PTRML', 'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL', 'NP3RTALL', 'NP3RTALJ', \
                       'NP3RTCON']
updrs3_total_col = 'NUPDRS3'
raw_mds_updrs3_df[updrs3_total_col] = raw_mds_updrs3_df[updrs3_symptom_cols].sum(axis=1)
'''
4 cases:
1. untreated: PD_MED_USE = 0
2. maob: PD_MED_USE = 3 (MAO-B inhibited, can't be defined as on/off)
3. on: PD_MED_USE not 0 or 3 AND (CMEDTM is not NaN OR ON_OFF_DOSE is 2)
4: off: PD_MED_USE is not 0 or 3 AND CMEDTM is NaN and ON_OFF_DOSE is not 2
'''
# very few PD_MED_USE == NaN - those will be thrown out
raw_mds_updrs3_df = raw_mds_updrs3_df.dropna(subset=['PD_MED_USE'])
# 4 patient-visits have CMEDTM despite PD_MED_USE = 0 -> TODO: handle this correctly, drop for now
raw_mds_updrs3_df = raw_mds_updrs3_df.loc[~np.logical_and(raw_mds_updrs3_df['PD_MED_USE']==0, \
                                                          ~pd.isnull(raw_mds_updrs3_df['CMEDTM']))]
raw_mds_updrs3_df_untreated = raw_mds_updrs3_df.loc[raw_mds_updrs3_df['PD_MED_USE']==0][standard3_cols + updrs3_symptom_cols \
                                                                                        + [updrs3_total_col]]
raw_mds_updrs3_df_maob = raw_mds_updrs3_df.loc[raw_mds_updrs3_df['PD_MED_USE']==3][standard3_cols + updrs3_symptom_cols \
                                                                                   + [updrs3_total_col]]
remaining_mds_updrs3_df = raw_mds_updrs3_df.loc[np.logical_and(raw_mds_updrs3_df['PD_MED_USE']!=0, \
                                                               raw_mds_updrs3_df['PD_MED_USE']!=3)]
raw_mds_updrs3_df_on \
    = remaining_mds_updrs3_df.loc[np.logical_or(~pd.isnull(remaining_mds_updrs3_df['CMEDTM']), \
                                                remaining_mds_updrs3_df['ON_OFF_DOSE']==2)][standard3_cols \
                                                                                            + updrs3_symptom_cols \
                                                                                            + [updrs3_total_col]]
raw_mds_updrs3_df_off \
    = remaining_mds_updrs3_df.loc[np.logical_and(pd.isnull(remaining_mds_updrs3_df['CMEDTM']), \
                                                 remaining_mds_updrs3_df['ON_OFF_DOSE']!=2)][standard3_cols \
                                                                                             + updrs3_symptom_cols \
                                                                                             + [updrs3_total_col]]
untreated_columns_map = dict()
untreated_cols = []
off_columns_map = dict()
off_cols = []
on_columns_map = dict()
on_cols = []
maob_columns_map = dict()
maob_cols = []
mean_columns_map = dict()
for col in updrs3_symptom_cols + ['NUPDRS3']:
    untreated_columns_map[col] = col + '_untreated'
    untreated_cols.append(col + '_untreated')
    off_columns_map[col] = col + '_off'
    off_cols.append(col + '_off')
    on_columns_map[col] = col + '_on'
    on_cols.append(col + '_on')
    maob_columns_map[col] = col + '_maob'
    maob_cols.append(col + '_maob')
    mean_columns_map[col] = {col: 'mean'}
raw_mds_updrs3_df_untreated = raw_mds_updrs3_df_untreated[standard3_cols + updrs3_symptom_cols + ['NUPDRS3']]
raw_mds_updrs3_df_untreated = raw_mds_updrs3_df_untreated.groupby(by=standard3_cols).agg(mean_columns_map)
raw_mds_updrs3_df_untreated.columns = raw_mds_updrs3_df_untreated.columns.droplevel(0)
raw_mds_updrs3_df_untreated = raw_mds_updrs3_df_untreated.reset_index()
raw_mds_updrs3_restruc_df = raw_mds_updrs3_df_untreated.rename(columns=untreated_columns_map)
raw_mds_updrs3_df_on = raw_mds_updrs3_df_on[standard3_cols + updrs3_symptom_cols + ['NUPDRS3']]
raw_mds_updrs3_df_on = raw_mds_updrs3_df_on.groupby(by=standard3_cols).agg(mean_columns_map)
raw_mds_updrs3_df_on.columns = raw_mds_updrs3_df_on.columns.droplevel(0)
raw_mds_updrs3_df_on = raw_mds_updrs3_df_on.reset_index()
raw_mds_updrs3_df_on.rename(columns=on_columns_map, inplace=True)
raw_mds_updrs3_df_off = raw_mds_updrs3_df_off[standard3_cols + updrs3_symptom_cols + ['NUPDRS3']]
raw_mds_updrs3_df_off = raw_mds_updrs3_df_off.groupby(by=standard3_cols).agg(mean_columns_map)
raw_mds_updrs3_df_off.columns = raw_mds_updrs3_df_off.columns.droplevel(0)
raw_mds_updrs3_df_off = raw_mds_updrs3_df_off.reset_index()
raw_mds_updrs3_df_off.rename(columns=off_columns_map, inplace=True)
raw_mds_updrs3_df_maob = raw_mds_updrs3_df_maob[standard3_cols + updrs3_symptom_cols + ['NUPDRS3']]
raw_mds_updrs3_df_maob = raw_mds_updrs3_df_maob.groupby(by=standard3_cols).agg(mean_columns_map)
raw_mds_updrs3_df_maob.columns = raw_mds_updrs3_df_maob.columns.droplevel(0)
raw_mds_updrs3_df_maob = raw_mds_updrs3_df_maob.reset_index()
raw_mds_updrs3_df_maob.rename(columns=maob_columns_map, inplace=True)
raw_mds_updrs3_restruc_df = raw_mds_updrs3_restruc_df.merge(raw_mds_updrs3_df_off, how = 'outer', validate = 'one_to_one')
raw_mds_updrs3_restruc_df = raw_mds_updrs3_restruc_df.merge(raw_mds_updrs3_df_on, how = 'outer', validate = 'one_to_one')
raw_mds_updrs3_restruc_df = raw_mds_updrs3_restruc_df.merge(raw_mds_updrs3_df_maob, how = 'outer', validate = 'one_to_one')
for col in raw_mds_updrs3_restruc_df:
    if col not in standard3_cols:
        raw_mds_updrs3_restruc_df[col] = raw_mds_updrs3_restruc_df[col].astype(np.float64)
if len(raw_mds_updrs3_restruc_df) != len(raw_mds_updrs3_restruc_df.drop_duplicates(subset=['PATNO','EVENT_ID'])):
    raw_mds_updrs3_restruc_df = average_by_patno_eventid(raw_mds_updrs3_restruc_df)
updrs3_total_cols = [untreated_cols[-1], off_cols[-1], on_cols[-1], maob_cols[-1]]
updrs3_symptom_cols = untreated_cols[:-1] + off_cols[:-1] + on_cols[:-1] + maob_cols[:-1]
symptom_cols += updrs3_symptom_cols
symptom_cols_dict['NUPDRS3'] = updrs3_symptom_cols
total_cols += updrs3_total_cols

# add NHY (Hoehn and Yahr)
raw_mds_updrs3_path = raw_datadir + 'MDS_UPDRS_Part_III.csv'
raw_mds_updrs3_df = pd.read_csv(raw_mds_updrs3_path)
raw_mds_updrs3_df['LAST_UPDATE'] = pd.to_datetime(raw_mds_updrs3_df['LAST_UPDATE'])
raw_mds_updrs3_df['INFODT'] = pd.to_datetime(raw_mds_updrs3_df['INFODT'])
raw_mds_updrs3_df = raw_mds_updrs3_df.sort_values(by=standard3_cols + ['LAST_UPDATE'])
nhy_df = raw_mds_updrs3_df[['PATNO','EVENT_ID','NHY']].drop_duplicates(subset=['PATNO','EVENT_ID'], keep='last')
raw_mds_updrs3_restruc_df = raw_mds_updrs3_restruc_df.merge(nhy_df, on=['PATNO','EVENT_ID'])
symptom_cols.append('NHY')

# MDS-UPDRS II
raw_mds_updrs2_path = raw_datadir + 'MDS_UPDRS_Part_II__Patient_Questionnaire.csv'
raw_mds_updrs2_df = pd.read_csv(raw_mds_updrs2_path)
updrs2_symptom_cols = ['NP2SPCH', 'NP2SALV', 'NP2SWAL', 'NP2EAT', 'NP2DRES', 'NP2HYGN', 'NP2HWRT', 'NP2HOBB', 'NP2TURN', \
                       'NP2TRMR', 'NP2RISE', 'NP2WALK','NP2FREZ']
updrs2_total_col = 'NUPDRS2'
raw_mds_updrs2_df[updrs2_total_col] = raw_mds_updrs2_df[updrs2_symptom_cols].sum(axis=1)
raw_mds_updrs2_df = raw_mds_updrs2_df[standard3_cols+updrs2_symptom_cols+[updrs2_total_col]]
for col in raw_mds_updrs2_df:
    if col not in standard3_cols:
        raw_mds_updrs2_df[col] = raw_mds_updrs2_df[col].astype(np.float64)
symptom_cols += updrs2_symptom_cols
symptom_cols_dict['NUPDRS2'] = updrs2_symptom_cols
total_cols.append(updrs2_total_col)
print('Merging MDS-UPDRS III + II')
collected_data = merge_dfs(raw_mds_updrs3_restruc_df, raw_mds_updrs2_df)

# Tremor or postural instability dominant
tremor_const_cols = ['NP2TRMR', 'NP3PTRMR', 'NP3PTRML', 'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL', 'NP3RTALL', \
                     'NP3RTALJ', 'NP3RTCON']
pigd_const_cols = ['NP2WALK', 'NP2FREZ', 'NP3GAIT', 'NP3FRZGT','NP3PSTBL']
tremor_const_untreated_cols = []
tremor_const_on_cols = []
tremor_const_off_cols = []
tremor_const_maob_cols = []
for col in tremor_const_cols:
    if col.startswith('NP3'):
        tremor_const_untreated_cols.append(col+'_untreated')
        tremor_const_on_cols.append(col+'_on')
        tremor_const_off_cols.append(col+'_off')
        tremor_const_maob_cols.append(col+'_maob')
    else:
        tremor_const_untreated_cols.append(col)
        tremor_const_on_cols.append(col)
        tremor_const_off_cols.append(col)
        tremor_const_maob_cols.append(col)
pigd_const_untreated_cols = []
pigd_const_on_cols = []
pigd_const_off_cols = []
pigd_const_maob_cols = []
for col in pigd_const_cols:
    if col.startswith('NP3'):
        pigd_const_untreated_cols.append(col+'_untreated')
        pigd_const_on_cols.append(col+'_on')
        pigd_const_off_cols.append(col+'_off')
        pigd_const_maob_cols.append(col+'_maob')
    else:
        pigd_const_untreated_cols.append(col)
        pigd_const_on_cols.append(col)
        pigd_const_off_cols.append(col)
        pigd_const_maob_cols.append(col)
collected_data['tremor_score_on'] = collected_data[tremor_const_on_cols].mean(axis=1)
collected_data['pigd_score_on'] = collected_data[pigd_const_on_cols].mean(axis=1)
collected_data['TD_PIGD_on'] = np.where(collected_data['pigd_score_on'] == 0, \
                                     np.where(collected_data['tremor_score_on'] == 0, 'indeterminate', 'TD'), \
                                     np.where(collected_data['tremor_score_on']/collected_data['pigd_score_on'] >= 1.15, 'TD', \
                                              np.where(collected_data['tremor_score_on']/collected_data['pigd_score_on'] <= .9, \
                                                       'PIGD', 'indeterminate')))
del collected_data['tremor_score_on']
del collected_data['pigd_score_on']
collected_data['TD_PIGD_on:tremor'] = np.where(collected_data['TD_PIGD_on']=='TD', 1, 0)
collected_data['TD_PIGD_on:posture'] = np.where(collected_data['TD_PIGD_on']=='PIGD', 1, 0)
collected_data['TD_PIGD_on:indet'] = np.where(collected_data['TD_PIGD_on']=='indeterminate', 1, 0)
del collected_data['TD_PIGD_on']
collected_data['tremor_score_off'] = collected_data[tremor_const_off_cols].mean(axis=1)
collected_data['pigd_score_off'] = collected_data[pigd_const_off_cols].mean(axis=1)
collected_data['TD_PIGD_off'] = np.where(collected_data['pigd_score_off'] == 0, \
                                     np.where(collected_data['tremor_score_off'] == 0, 'indeterminate', 'TD'), \
                                     np.where(collected_data['tremor_score_off']/collected_data['pigd_score_off'] >= 1.15, 'TD', \
                                              np.where(collected_data['tremor_score_off']/collected_data['pigd_score_off'] <= .9, \
                                                       'PIGD', 'indeterminate')))
del collected_data['tremor_score_off']
del collected_data['pigd_score_off']
collected_data['TD_PIGD_off:tremor'] = np.where(collected_data['TD_PIGD_off']=='TD', 1, 0)
collected_data['TD_PIGD_off:posture'] = np.where(collected_data['TD_PIGD_off']=='PIGD', 1, 0)
collected_data['TD_PIGD_off:indet'] = np.where(collected_data['TD_PIGD_off']=='indeterminate', 1, 0)
del collected_data['TD_PIGD_off']
collected_data['tremor_score_untreated'] = collected_data[tremor_const_on_cols].mean(axis=1)
collected_data['pigd_score_untreated'] = collected_data[pigd_const_on_cols].mean(axis=1)
collected_data['TD_PIGD_untreated'] = np.where(collected_data['pigd_score_untreated'] == 0, \
                                     np.where(collected_data['tremor_score_untreated'] == 0, 'indeterminate', 'TD'), \
                                     np.where(collected_data['tremor_score_untreated']/collected_data['pigd_score_untreated'] >= 1.15, 'TD', \
                                              np.where(collected_data['tremor_score_untreated']/collected_data['pigd_score_untreated'] <= .9, \
                                                       'PIGD', 'indeterminate')))
del collected_data['tremor_score_untreated']
del collected_data['pigd_score_untreated']
collected_data['TD_PIGD_untreated:tremor'] = np.where(collected_data['TD_PIGD_untreated']=='TD', 1, 0)
collected_data['TD_PIGD_untreated:posture'] = np.where(collected_data['TD_PIGD_untreated']=='PIGD', 1, 0)
collected_data['TD_PIGD_untreated:indet'] = np.where(collected_data['TD_PIGD_untreated']=='indeterminate', 1, 0)
del collected_data['TD_PIGD_untreated']
collected_data['tremor_score_maob'] = collected_data[tremor_const_on_cols].mean(axis=1)
collected_data['pigd_score_maob'] = collected_data[pigd_const_on_cols].mean(axis=1)
collected_data['TD_PIGD_maob'] = np.where(collected_data['pigd_score_maob'] == 0, \
                                     np.where(collected_data['tremor_score_maob'] == 0, 'indeterminate', 'TD'), \
                                     np.where(collected_data['tremor_score_maob']/collected_data['pigd_score_maob'] >= 1.15, \
                                              'TD', \
                                              np.where(collected_data['tremor_score_maob']/collected_data['pigd_score_maob'] \
                                                       <= .9, 'PIGD', 'indeterminate')))
del collected_data['tremor_score_maob']
del collected_data['pigd_score_maob']
collected_data['TD_PIGD_maob:tremor'] = np.where(collected_data['TD_PIGD_maob']=='TD', 1, 0)
collected_data['TD_PIGD_maob:posture'] = np.where(collected_data['TD_PIGD_maob']=='PIGD', 1, 0)
collected_data['TD_PIGD_maob:indet'] = np.where(collected_data['TD_PIGD_maob']=='indeterminate', 1, 0)
del collected_data['TD_PIGD_maob']
td_pigd_cols = ['TD_PIGD_on:tremor', 'TD_PIGD_on:posture', 'TD_PIGD_on:indet', 'TD_PIGD_off:tremor', 'TD_PIGD_off:posture', 'TD_PIGD_off:indet', 'TD_PIGD_untreated:tremor', 'TD_PIGD_untreated:posture', 'TD_PIGD_untreated:indet', 'TD_PIGD_maob:tremor', 'TD_PIGD_maob:posture', 'TD_PIGD_maob:indet']
for col in td_pigd_cols:
    collected_data[col] = collected_data[col].astype(np.float64)
shared_cols += td_pigd_cols
shared_cols_dict['TD_PIGD'] = td_pigd_cols

# Subject center number
cno_path = raw_datadir + 'Center-Subject_List.csv'
cno_df = pd.read_csv(cno_path)
collected_data = collected_data.merge(cno_df[['PATNO','CNO']],on='PATNO',how='outer',validate='many_to_one')
collected_data['CNO'] = collected_data['CNO'].astype(np.float64)
baseline_cols.append('CNO')
baseline_cols_dict['CNO'] = ['CNO']

# Age
random_path = raw_datadir + 'Randomization_table.csv'
random_df = pd.read_csv(random_path)
collected_data = collected_data.merge(random_df[['PATNO','BIRTHDT']], on='PATNO', how='outer')

# Gender
collected_data = collected_data.merge(random_df[['PATNO','GENDER']], on='PATNO', how='outer')
collected_data['MALE'] = np.where(collected_data['GENDER']==2, 1, 0)
#collected_data['FEMALE'] = np.where(collected_data['GENDER']==2, 0, 1)
del collected_data['GENDER']
gender_cols = ['MALE']#, 'FEMALE']
baseline_cols += gender_cols
baseline_cols_dict['GENDER'] = gender_cols
for col in gender_cols:
    collected_data[col] = collected_data[col].astype(np.float64)

# Race
screening_path = raw_datadir + 'Screening___Demographics.csv'
screening_df = pd.read_csv(screening_path)
race_cols = ['HISPLAT','RAWHITE', 'RAASIAN', 'RABLACK', 'RAINDALS', 'RAHAWOPI', 'RANOS']
for col in race_cols:
    screening_df[col] = np.where(screening_df[col]==2, 0, screening_df[col]) # 2 = unknown/not reported
consent_col = 'CONSNTDT'
collected_data = collected_data.merge(screening_df[['PATNO',consent_col]+race_cols], on='PATNO', how='outer')
baseline_cols += race_cols
baseline_cols_dict['RACE'] = race_cols
for col in race_cols:
    collected_data[col] = collected_data[col].astype(np.float64)

# Family history
famhist_path = raw_datadir + 'Family_History__PD_.csv'
famhist_df = pd.read_csv(famhist_path)
famhist_df = famhist_df.drop_duplicates(subset='PATNO', keep='last')
famhist_cols = ['BIOMOMPD', 'BIODADPD', 'FULSIBPD', 'HAFSIBPD', 'MAGPARPD', 'PAGPARPD', 'MATAUPD', 'PATAUPD', 'KIDSPD']
collected_data = collected_data.merge(famhist_df[['PATNO']+famhist_cols], on='PATNO', how='outer')
baseline_cols += famhist_cols
baseline_cols_dict['FAMHIST'] = famhist_cols
for col in famhist_cols:
    collected_data[col] = collected_data[col].astype(np.float64)

# Dominant side affected by PD, diagnosis date
pd_feat_path = raw_datadir + 'PD_Features.csv'
pd_feat_df = pd.read_csv(pd_feat_path)
pd_feat_df['RIGHT_DOMSIDE'] = np.where(pd_feat_df['DOMSIDE'].astype(str)=='1',1,0)
pd_feat_df['LEFT_DOMSIDE'] = np.where(pd_feat_df['DOMSIDE'].astype(str)=='2',1,0)
#pd_feat_df['MIXED_DOMSIDE'] = np.where(pd_feat_df['DOMSIDE'].astype(str)=='3',1,0)
pd_feat_cols = ['RIGHT_DOMSIDE', 'LEFT_DOMSIDE', 'PDDXDT'] #'MIXED_DOMSIDE', 
collected_data = collected_data.merge(pd_feat_df[['PATNO']+pd_feat_cols], on='PATNO', how='outer', validate='many_to_one')
shared_cols += pd_feat_cols
shared_cols_dict['PD_DOMSIDE'] = ['RIGHT_DOMSIDE', 'LEFT_DOMSIDE']#, 'MIXED_DOMSIDE']
for col in ['RIGHT_DOMSIDE', 'LEFT_DOMSIDE']: #, 'MIXED_DOMSIDE']:
    collected_data[col] = collected_data[col].astype(np.float64)

# prodromal diagnostic questionnaire
prodiagq_df = pd.read_csv(raw_datadir + 'Prodromal_Diagnostic_Questionnaire.csv')
prodiagq_restruc_df = prodiagq_df[['PATNO','EVENT_ID','PRIMDIAG']].drop_duplicates()
prodiagq_restruc_df['PRODROMAL_DIAG:PHENOCONV'] = np.where(prodiagq_restruc_df['PRIMDIAG'] == 1, 1, 0)
prodiagq_restruc_df['PRODROMAL_DIAG:NONMOTOR_PRODROMA'] = np.where(prodiagq_restruc_df['PRIMDIAG'] == 23, 1, 0)
prodiagq_restruc_df['PRODROMAL_DIAG:MOTOR_PRODROMA'] = np.where(prodiagq_restruc_df['PRIMDIAG'] == 24, 1, 0)
prodiagq_restruc_df['PRODROMAL_DIAG:NO_NEURO'] = np.where(prodiagq_restruc_df['PRIMDIAG'] == 17, 1, 0)
del prodiagq_restruc_df['PRIMDIAG']
collected_data = collected_data.merge(prodiagq_restruc_df, on=['PATNO','EVENT_ID'], how='outer', validate='many_to_one')
prodromal_diag_cols = ['PRODROMAL_DIAG:PHENOCONV', 'PRODROMAL_DIAG:NONMOTOR_PRODROMA', 'PRODROMAL_DIAG:MOTOR_PRODROMA', 'PRODROMAL_DIAG:NO_NEURO']
shared_cols += prodromal_diag_cols
eventid_only_cols += prodromal_diag_cols
shared_cols_dict['PRODROMAL_DIAG'] = prodromal_diag_cols
for col in prodromal_diag_cols:
    collected_data[col] = collected_data[col].astype(np.float64)

# REM sleep
rem_sleep_path = raw_datadir + 'REM_Sleep_Disorder_Questionnaire.csv'
rem_sleep_df = pd.read_csv(rem_sleep_path)
rem_symptom1_cols = ['DRMVIVID', 'DRMAGRAC', 'DRMNOCTB', 'SLPLMBMV', 'SLPINJUR', 'DRMVERBL', 'DRMFIGHT', 'DRMUMV', \
                     'DRMOBJFL', 'MVAWAKEN', 'DRMREMEM', 'SLPDSTRB']
rem_symptom2_cols = ['STROKE', 'HETRA', 'PARKISM', 'RLS', 'NARCLPSY', 'DEPRS', 'EPILEPSY', 'BRNINFM', 'CNSOTH']
rem_sleep_df['first_sum'] = rem_sleep_df[rem_symptom1_cols].sum(axis=1)
rem_sleep_df['second_one'] = np.where(rem_sleep_df[rem_symptom2_cols].sum(axis=1) > 0, 1, 0)
rem_total_col = 'REMSLEEP'
rem_sleep_df[rem_total_col] = np.where(rem_sleep_df[['first_sum', 'second_one']].sum(axis=1) >= 5, 1, 0)
print('Adding REM sleep')
all_rem_sleep_cols = rem_symptom1_cols + rem_symptom2_cols + [rem_total_col]
for col in all_rem_sleep_cols:
    rem_sleep_df[col] = rem_sleep_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, rem_sleep_df[standard3_cols + all_rem_sleep_cols])
symptom_cols += rem_symptom1_cols + rem_symptom2_cols
symptom_cols_dict['REMSLEEP'] = rem_symptom1_cols + rem_symptom2_cols
total_cols.append(rem_total_col)

# Education and dominant hand
socio_eco_path = raw_datadir + 'Socio-Economics.csv'
socio_eco_df = pd.read_csv(socio_eco_path)
socio_eco_df = socio_eco_df.drop_duplicates(subset='PATNO', keep='last')
#socio_eco_df['EDUCYRS_opp'] = socio_eco_df['EDUCYRS'].max() - socio_eco_df['EDUCYRS']
socio_eco_df['RIGHT_HANDED'] = np.where(socio_eco_df['HANDED']==1,1,0)
socio_eco_df['LEFT_HANDED'] = np.where(socio_eco_df['HANDED']==2,1,0)
#socio_eco_df['MIXED_HANDED'] = np.where(socio_eco_df['HANDED']==3,1,0)
socio_eco_cols = ['EDUCYRS', 'RIGHT_HANDED', 'LEFT_HANDED']#, 'MIXED_HANDED']
for col in socio_eco_cols:
    socio_eco_df[col] = socio_eco_df[col].astype(np.float64)
collected_data = collected_data.merge(socio_eco_df[['PATNO'] + socio_eco_cols], on='PATNO', how='outer', validate='many_to_one')
baseline_cols += socio_eco_cols
baseline_cols_dict['EDUCYRS'] = ['EDUCYRS']
baseline_cols_dict['HANDED'] = ['RIGHT_HANDED', 'LEFT_HANDED']#, 'MIXED_HANDED']

# get enroll category for datscan processing
patient_status_path = raw_datadir + 'Patient_Status.csv'
patient_status_df = pd.read_csv(patient_status_path)

# Imaging
datscan_path = raw_datadir + 'DATScan_Analysis.csv'
datscan_df = pd.read_csv(datscan_path)
datscan_df = datscan_df.drop_duplicates(subset=['PATNO','EVENT_ID'])
domsides_df = collected_data[standard3_cols + ['LEFT_DOMSIDE','RIGHT_DOMSIDE']]
domsides_df = domsides_df.sort_values(by=['INFODT'])
domsides_df = domsides_df.drop_duplicates(subset=['PATNO','EVENT_ID'])
datscan_df = datscan_df.merge(domsides_df[['PATNO','EVENT_ID','LEFT_DOMSIDE','RIGHT_DOMSIDE']], \
                              on=['PATNO','EVENT_ID'], validate='one_to_one')
datscan_df = datscan_df.merge(patient_status_df[['PATNO','ENROLL_CAT']], validate='many_to_one')
datscan_df['caudate_mean'] = datscan_df[['CAUDATE_L','CAUDATE_R']].mean(axis=1)
datscan_df['putamen_mean'] = datscan_df[['PUTAMEN_L','PUTAMEN_R']].mean(axis=1)
datscan_df['contralateral_caudate'] = np.where(datscan_df['ENROLL_CAT'].isin({'PD','SWEDD','GENPD','REGPD'}), \
                                               np.where(datscan_df['LEFT_DOMSIDE'] == 1, datscan_df['CAUDATE_L'], \
                                                        np.where(datscan_df['RIGHT_DOMSIDE'] == 1, datscan_df['CAUDATE_R'], \
                                                                 datscan_df['caudate_mean'])), \
                                               datscan_df['caudate_mean'])
datscan_df['ipsilateral_caudate'] = np.where(datscan_df['ENROLL_CAT'].isin({'PD','SWEDD','GENPD','REGPD'}), \
                                             np.where(datscan_df['RIGHT_DOMSIDE'] == 1, datscan_df['CAUDATE_L'], \
                                                      np.where(datscan_df['LEFT_DOMSIDE'] == 1, datscan_df['CAUDATE_R'], \
                                                               datscan_df['caudate_mean'])), \
                                             datscan_df['caudate_mean'])
datscan_df['contralateral_putamen'] = np.where(datscan_df['ENROLL_CAT'].isin({'PD','SWEDD','GENPD','REGPD'}), \
                                               np.where(datscan_df['LEFT_DOMSIDE'] == 1, datscan_df['PUTAMEN_L'], \
                                                        np.where(datscan_df['RIGHT_DOMSIDE'] == 1, datscan_df['PUTAMEN_R'], \
                                                                 datscan_df['putamen_mean'])), \
                                               datscan_df['putamen_mean'])
datscan_df['ipsilateral_putamen'] = np.where(datscan_df['ENROLL_CAT'].isin({'PD','SWEDD','GENPD','REGPD'}), \
                                             np.where(datscan_df['RIGHT_DOMSIDE'] == 1, datscan_df['PUTAMEN_L'], \
                                                      np.where(datscan_df['LEFT_DOMSIDE'] == 1, datscan_df['PUTAMEN_R'], \
                                                               datscan_df['putamen_mean'])), \
                                             datscan_df['putamen_mean'])
datscan_df['count_density_ratio_L'] = datscan_df['CAUDATE_L']/datscan_df['PUTAMEN_L']
datscan_df['count_density_ratio_R'] = datscan_df['CAUDATE_R']/datscan_df['PUTAMEN_R']
datscan_df['count_density_ratio_mean'] = datscan_df[['count_density_ratio_L','count_density_ratio_R']].mean(axis=1)
datscan_df['count_density_ratio_contralateral'] = np.where(datscan_df['ENROLL_CAT'].isin({'PD','SWEDD','GENPD','REGPD'}), \
                                                           np.where(datscan_df['LEFT_DOMSIDE'] == 1, \
                                                                    datscan_df['count_density_ratio_L'], \
                                                        np.where(datscan_df['RIGHT_DOMSIDE'] == 1, \
                                                                 datscan_df['count_density_ratio_R'], \
                                                                 datscan_df['count_density_ratio_mean'])), \
                                               datscan_df['count_density_ratio_mean'])
datscan_df['count_density_ratio_ipsilateral'] = np.where(datscan_df['ENROLL_CAT'].isin({'PD','SWEDD','GENPD','REGPD'}), \
                                                           np.where(datscan_df['RIGHT_DOMSIDE'] == 1, \
                                                                    datscan_df['count_density_ratio_L'], \
                                                        np.where(datscan_df['LEFT_DOMSIDE'] == 1, \
                                                                 datscan_df['count_density_ratio_R'], \
                                                                 datscan_df['count_density_ratio_mean'])), \
                                               datscan_df['count_density_ratio_mean'])
datscan_df['asymmetry_index_caudate'] = (100*(datscan_df['CAUDATE_L']-datscan_df['CAUDATE_R'])/datscan_df['caudate_mean']).abs()
datscan_df['asymmetry_index_putamen'] = (100*(datscan_df['PUTAMEN_L']-datscan_df['PUTAMEN_R'])/datscan_df['putamen_mean']).abs()
datscan_cols = ['contralateral_caudate', 'ipsilateral_caudate', 'contralateral_putamen', 'ipsilateral_putamen', \
                'count_density_ratio_contralateral', 'count_density_ratio_ipsilateral', 'asymmetry_index_caudate', \
                'asymmetry_index_putamen']
for col in datscan_cols:
    datscan_df[col] = datscan_df[col].astype(np.float64)
collected_data = collected_data.merge(datscan_df[['PATNO','EVENT_ID'] + datscan_cols], on = ['PATNO','EVENT_ID'], how='outer', validate='many_to_one')
shared_cols += datscan_cols
eventid_only_cols += datscan_cols
shared_cols_dict['DATSCAN'] = datscan_cols

# Biospecimens, including CSFs, SNPs, RNA expression, and biochemical labs
def get_biospec_cols(biospec_df, cols, use_infodt=True):
    if use_infodt:
        cols_df = biospec_df[standard3_cols].drop_duplicates()
    else:
        cols_df = biospec_df[['PATNO','EVENT_ID']].drop_duplicates()
    for col in cols:
        specific_biospec_df = biospec_df.loc[biospec_df['TESTNAME']==col]
        if use_infodt:
            cols_df = cols_df.merge(specific_biospec_df[standard3_cols + ['TESTVALUE']], on=standard3_cols, \
                                    how='left', validate='one_to_one')
        else:
            cols_df = cols_df.merge(specific_biospec_df[['PATNO','EVENT_ID','TESTVALUE']], on=['PATNO','EVENT_ID'], \
                                    how='left', validate='one_to_one')
        cols_df.rename(columns={'TESTVALUE': col}, inplace=True)
    return cols_df.dropna(how='all', subset=cols)

biospec_path = raw_datadir + 'Current_Biospecimen_Analysis_Results.csv'
biospec_df = pd.read_csv(biospec_path)
biospec_df = biospec_df.sort_values(by=['RUNDATE', 'update_stamp'])
biospec_df.rename(columns={'CLINICAL_EVENT': 'EVENT_ID', 'RUNDATE': 'INFODT'}, inplace=True)
biospec_df = biospec_df.loc[~biospec_df['UNITS'].isin({'SD','Stdev','Std. Error of Mean'})]
biospec_df = biospec_df.drop_duplicates(subset=standard3_cols + ['TESTNAME'], keep='last')
biospec_df['TESTVALUE'] = np.where(biospec_df['TESTVALUE'].astype(str).str.contains('<'), 0, biospec_df['TESTVALUE'])
biospec_df['TESTVALUE'] = np.where(biospec_df['TESTVALUE'].astype(str).str.contains('below'), 0, biospec_df['TESTVALUE'])
biospec_df['TESTVALUE'] = np.where(np.logical_and(biospec_df['TESTVALUE'].astype(str) == '>ULOQ', \
                                                  biospec_df['TESTNAME'] == '3-Methoxytyrosine'), \
                                   288, biospec_df['TESTVALUE']) # 277.8 was observed max
biospec_df['TESTVALUE'] = np.where(np.logical_and(biospec_df['TESTVALUE'].astype(str) == '>ULOQ', \
                                                  biospec_df['TESTNAME'] == '3,4-Dihydroxyphenylalanine (DOPA)'), \
                                   19.5, biospec_df['TESTVALUE']) # 19.3 was observed max
biospec_df['TESTVALUE'] = np.where(np.logical_or(biospec_df['TESTVALUE'].astype(str) == '>12500 ng/ml', \
                                                 biospec_df['TESTVALUE'].astype(str) == '>12500ng/ml'), \
                                   12500, biospec_df['TESTVALUE'])
biospec_df['TESTVALUE'] = np.where(np.logical_and(biospec_df['TESTVALUE'].astype(str) == 'above', \
                                                  biospec_df['TESTNAME'] == 'CSF Hemoglobin'), \
                                   12500, biospec_df['TESTVALUE'])
biospec_df['TESTVALUE'] = np.where(biospec_df['TESTVALUE'].astype(str) == '>20', 20, biospec_df['TESTVALUE']) # odd since higher measurements exist
biospec_df['TESTVALUE'] = np.where(biospec_df['TESTVALUE'].astype(str).str.lower() == 'undetermined', np.nan, biospec_df['TESTVALUE'])
csf_orig_cols = ['pTau', 'tTau', 'ABeta 1-42', 'CSF Alpha-synuclein']
csf_df = get_biospec_cols(biospec_df, csf_orig_cols)
for col in csf_orig_cols:
    csf_df[col] = csf_df[col].astype(np.float64)
#csf_df['ABeta 1-42'] = np.where(csf_df['ABeta 1-42']=='<200', 0, csf_df['ABeta 1-42']).astype(np.float64)
#csf_df['tTau'] = np.where(csf_df['tTau']=='<80', 0, csf_df['tTau']).astype(np.float64)
#csf_df['pTau'] = np.where(csf_df['pTau']=='<8', 0, csf_df['pTau']).astype(np.float64)
csf_df['ABETA_log'] = np.log(csf_df['ABeta 1-42']+0.01) # to avoid -inf. Because most in the 100's scale, won't change much
csf_df['TTAU_log'] = np.log(csf_df['tTau']+0.01)
csf_df['PTAU_log'] = np.log(csf_df['pTau']+0.01)
csf_df['ASYNU_log'] = np.log(csf_df['CSF Alpha-synuclein']+0.01)
csf_df['PTAU_ABETA_ratio'] = csf_df['pTau']/(csf_df['ABeta 1-42']+0.01) # to avoid nan. those will be large though
csf_df['TTAU_ABETA_ratio'] = csf_df['tTau']/(csf_df['ABeta 1-42']+0.01)
csf_df['PTAU_TTAU_ratio'] = csf_df['pTau']/(csf_df['tTau']+0.01)
del csf_df['ABeta 1-42']
del csf_df['tTau']
del csf_df['pTau']
csf_cols = ['ABETA_log', 'TTAU_log', 'PTAU_log', 'ASYNU_log', 'PTAU_ABETA_ratio', 'TTAU_ABETA_ratio', 'PTAU_TTAU_ratio']
print('Adding CSF')
collected_data = merge_dfs(collected_data, csf_df[standard3_cols + csf_cols])
shared_cols += csf_cols
shared_cols_dict['CSF'] = csf_cols

# gather genetic risk score
genetic_score_df = biospec_df.loc[biospec_df['TESTNAME']=='SCORE']
genetic_score_df.rename(columns={'TESTVALUE': 'GENETIC_RISK_SCORE'}, inplace=True)
collected_data = merge_dfs(collected_data, genetic_score_df[standard3_cols + ['GENETIC_RISK_SCORE']])
shared_cols.append('GENETIC_RISK_SCORE')
shared_cols_dict['GENETIC_RISK_SCORE'] = ['GENETIC_RISK_SCORE']

def make_snps_onehot(snp_df, snp_cols):
    # returns one-hot dataframe and list of one-hot columns
    onehot_snp_df = snp_df[standard3_cols]
    onehot_snp_cols = []
    for col in snp_cols:
        snp_vals = snp_df[col].unique()
        for val in snp_vals:
            if str(val) != 'nan': 
                col_val = str(col)+'_'+str(val)
                onehot_snp_cols.append(col_val)
                onehot_snp_df[col_val] = np.where(pd.isnull(snp_df[col]), snp_df[col], np.where(snp_df[col]==val, 1, 0))
    return onehot_snp_df, onehot_snp_cols

biospec_df['TESTNAME'] = np.where(biospec_df['TESTNAME'].str.lower()=='apoe genotype', 'ApoE Genotype', \
                                  biospec_df['TESTNAME'])
snp_cols = ['rs823118', 'rs3910105', 'rs356181', 'rs55785911', 'rs2414739', 'rs329648', 'rs11724635', 'rs17649553', \
            'rs114138760', 'ApoE Genotype', 'rs11868035', 'rs71628662', 'rs118117788', 'rs11158026', 'rs34884217', \
            'rs34311866', 'rs199347', 'rs6430538', 'rs34995376_LRRK2_p.R1441H', 'rs11060180', 'rs76763715_GBA_p.N370S', \
            'rs12637471', 'rs8192591', 'rs12456492', 'rs14235', 'rs35801418_LRRK2_p.Y1699C', 'rs591323', 'rs6812193', \
            'rs76904798', 'rs34637584_LRRK2_p.G2019S', 'rs10797576', 'rs115462410', 'rs1955337', 'rs35870237_LRRK2_p.I2020T']
snp_df = get_biospec_cols(biospec_df, snp_cols)
snp_onehot_df, snp_onehot_cols = make_snps_onehot(snp_df, snp_cols)
print('Adding SNPs')
snp_onehot_df = snp_onehot_df.sort_values(by=['EVENT_ID'])
snp_onehot_df = snp_onehot_df.drop_duplicates(subset=['PATNO'])
snp_onehot_df['EVENT_ID'] = 'SC'
for col in snp_onehot_cols:
    snp_onehot_df[col] = snp_onehot_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, snp_onehot_df)
screening_cols += snp_onehot_cols
screening_cols_dict['SNP'] = snp_onehot_cols

# gather more SNPs from WGS datafile
wgs_df = pd.read_csv(raw_datadir + 'PPMI_PD_Variants_Genetic_Status_WGS_20180921.csv')
biospec_snps = []
for val in biospec_df.TESTNAME.unique():
    if val.startswith('rs'):
        biospec_snps.append(val)
modified_biospec_snps = dict() # map rs... to original test name
for snp in biospec_snps[1:]:
    if '_' in snp:
        modified_biospec_snps[snp[:snp.index('_')]] = snp
    else:
        modified_biospec_snps[snp] = snp
wgs_to_testname_map = dict()
for col in wgs_df.columns.values:
    rs_part = col[col.rfind('_')+1:]
    if rs_part in modified_biospec_snps.keys():
        wgs_to_testname_map[col] = modified_biospec_snps[rs_part]
wgs_df_reformatted = wgs_df[['PATNO']]
for wgs_col in wgs_df.columns:
    if wgs_col == 'PATNO':
        continue
    # count is minor allele after _, typical format: chr1:154925709:G:C_C_PMVK_rs114138760
    after_first_colon = wgs_col[wgs_col.index(':')+1:]
    after_second_colon = after_first_colon[after_first_colon.index(':')+1:]
    first_allele = after_second_colon[:after_second_colon.index(':')]
    second_allele = after_second_colon[after_second_colon.index(':')+1:after_second_colon.index('_')]
    after_first_underscore = wgs_col[wgs_col.index('_')+1:]
    counted_allele = after_first_underscore[:after_first_underscore.index('_')]
    if counted_allele == first_allele:
        other_allele = second_allele
    else:
        other_allele = first_allele
    # match biospecimen format if exists
    rs_str = wgs_col[wgs_col.rindex('_')+1:]
    counted_1_str = None
    biospec_allele_presence = {0: False, 1: False, 2: False}
    if wgs_col in wgs_to_testname_map.keys():
        testname = wgs_to_testname_map[wgs_col]
        testvalues = biospec_df.loc[biospec_df['TESTNAME']==testname].dropna(subset=['TESTVALUE']).TESTVALUE.unique()
        test_val_alleles = set()
        for test_val in testvalues:
            first_test_val_allele = test_val[:test_val.index('/')]
            test_val_alleles.add(first_test_val_allele)
            second_test_val_allele = test_val[test_val.index('/')+1:]
            test_val_alleles.add(second_test_val_allele)
            if first_test_val_allele != second_test_val_allele:
                counted_1_str = rs_str + '_' + first_test_val_allele + '/' + second_test_val_allele
                biospec_allele_presence[1] = True
        if counted_allele not in test_val_alleles and other_allele not in test_val_alleles:
            antisense_map = {'A': 'T', 'C': 'G', 'T': 'A', 'G': 'C'}
            counted_allele = antisense_map[counted_allele]
            other_allele = antisense_map[other_allele]
        if other_allele + '/' + other_allele in testvalues:
            biospec_allele_presence[0] = True
        if counted_allele + '/' + counted_allele in testvalues:
            biospec_allele_presence[2] = True
    counted_0_str = rs_str + '_' + other_allele + '/' + other_allele
    if counted_1_str is None:
        counted_1_str = rs_str + '_' + other_allele + '/' + counted_allele
    counted_2_str = rs_str + '_' + counted_allele + '/' + counted_allele
    wgs_df_reformatted[counted_0_str] = np.where(wgs_df[wgs_col]==0, 1, 0)
    wgs_df_reformatted[counted_1_str] = np.where(wgs_df[wgs_col]==1, 1, 0)
    wgs_df_reformatted[counted_2_str] = np.where(wgs_df[wgs_col]==2, 1, 0)
    # if combination doesn't exist in wgs or biospec, then remove
    if wgs_df_reformatted[counted_0_str].sum() == 0 and not biospec_allele_presence[0]:
        del wgs_df_reformatted[counted_0_str]
    if wgs_df_reformatted[counted_1_str].sum() == 0 and not biospec_allele_presence[1]:
        del wgs_df_reformatted[counted_1_str]
    if wgs_df_reformatted[counted_2_str].sum() == 0 and not biospec_allele_presence[2]:
        del wgs_df_reformatted[counted_2_str]
# update collected_data for shared columns first, then merge rest
wgs_reformatted_cols = wgs_df_reformatted.columns.values[1:]
wgs_df_reformatted['EVENT_ID'] = 'SC'
for col in wgs_reformatted_cols:
    if col in collected_data:
        wgs_df_reformatted.rename(columns={col: col + '_wgs'}, inplace=True)
        collected_data = collected_data.merge(wgs_df_reformatted[['PATNO', 'EVENT_ID', col + '_wgs']], \
                                              on=['PATNO', 'EVENT_ID'], how='outer', validate='many_to_one')
        collected_data[col] = np.where(pd.isnull(collected_data[col]), collected_data[col + '_wgs'], collected_data[col])
        del wgs_df_reformatted[col + '_wgs']
collected_data = collected_data.merge(wgs_df_reformatted, on=['PATNO','EVENT_ID'], how='outer', validate='many_to_one')
remaining_wgs_cols = wgs_df_reformatted.columns.values[1:-1].tolist()
screening_cols += remaining_wgs_cols
screening_cols_dict['SNP'] += remaining_wgs_cols
eventid_only_cols += remaining_wgs_cols

# get healthy control cohort for normalization below
patient_status_path = raw_datadir + 'Patient_Status.csv'
patient_status_df = pd.read_csv(patient_status_path)
hc_patnos = patient_status_df.loc[patient_status_df['ENROLL_CAT']=='HC'].PATNO.unique()

# RNA seq normalization
# take average of housekeeping genes
# subtract average from gene of interest -> delta Ct
# take average of delta Ct across healthy controls for each gene
# subtract above average from delta Ct
def get_normalized_rna(rna_df, feat_cols, housekeeping_cols):
    # if no housekeeping results for a patient, remove b/c not usable
    rna_df = rna_df.dropna(subset=housekeeping_cols, how='all')
    if len(rna_df) == 0:
        print('Housekeeping genes never measured')
        return None
    rna_df['housekeeping_mean'] = np.nanmean(rna_df[housekeeping_cols].astype(np.float64), axis=1)
    for col in feat_cols:
        rna_df[col] = rna_df[col].astype(np.float64) - rna_df['housekeeping_mean'].astype(np.float64)
    hc_df = rna_df.loc[rna_df['PATNO'].isin(hc_patnos)].dropna(subset=feat_cols, how='all')
    if len(hc_df) == 0:
        print('No healthy controls measured')
        return None
    #hc_avg_delta_df = np.nanmean(rna_df.loc[rna_df['PATNO'].isin(hc_patnos)][feat_cols], axis=0)
    num_cols_removed = 0
    for col in feat_cols:
        hc_col_df = hc_df[col].dropna()
        if len(hc_col_df) == 0:
            del rna_df[col]
            num_cols_removed += 1
        else:
            rna_df[col] = rna_df[col].astype(np.float64) - np.nanmean(hc_col_df.astype(np.float64))
    if num_cols_removed == len(feat_cols):
        print('No healthy controls measured 2')
        return None
    return rna_df

avg_cycle_thresh_cols = ['COPZ1', 'WLS', 'SOD2', 'APP', 'ZNF160', 'HNF4A', 'GAPDH', 'EFTUD2', 'PTBP1', 'C5ORF4']
avg_cycle_thresh_housekeeping_cols = ['GAPDH']
avg_cycle_thresh_feature_cols = ['COPZ1', 'ZNF160', 'PTBP1', 'C5ORF4', 'WLS', 'SOD2', 'APP', 'HNF4A', 'EFTUD2']
avg_cycle_thresh_df = get_biospec_cols(biospec_df, avg_cycle_thresh_cols, use_infodt=False)
avg_cycle_thresh_df = get_normalized_rna(avg_cycle_thresh_df, avg_cycle_thresh_feature_cols, avg_cycle_thresh_housekeeping_cols)
if avg_cycle_thresh_df is None:
    print('No RNA 1 to add to data')
    rna_cols = []
else:
    print('Adding RNA 1')
    for col in avg_cycle_thresh_feature_cols:
        avg_cycle_thresh_df[col] = avg_cycle_thresh_df[col].astype(np.float64)
    collected_data = collected_data.merge(avg_cycle_thresh_df[['PATNO','EVENT_ID'] + avg_cycle_thresh_feature_cols], \
                                          on=['PATNO','EVENT_ID'], how='outer', validate='one_to_one')
    #collected_data = merge_dfs(collected_data, avg_cycle_thresh_df[standard3_cols + avg_cycle_thresh_feature_cols])
    shared_cols += avg_cycle_thresh_feature_cols
    rna_cols = avg_cycle_thresh_feature_cols

def avg_reps(rna_df):
    # returns dataframe with average across reps and list of feature names
    rna_avg_df = rna_df[standard3_cols]
    rna_cols = []
    for col in rna_df:
        if 'rep 1' in col:
            avg_col = col[:col.index(' ')]
            rep2_col = col.replace('rep 1','rep 2')
            rna_avg_df[avg_col] = np.nanmean(rna_df[[col, rep2_col]].astype(np.float64), axis=1)
            rna_cols.append(avg_col)
    return rna_avg_df, rna_cols

# same as above but take arithmetic mean of reps first
rep_cycle_thresh_cols = ['ALDH1A1 (rep 1)', 'UBE2K (rep 2)', 'GAPDH (rep 1)', 'ALDH1A1 (rep 2)', 'HSPA8 (rep 1)', \
                         'UBE2K (rep 1)', 'PGK1 (rep 1)', 'LAMB2 (rep 2)', 'GAPDH (rep 2)', 'SKP1 (rep 2)', 'SKP1 (rep 1)', \
                         'LAMB2 (rep 1)', 'PGK1 (rep 2)', 'HSPA8 (rep 2)', 'PSMC4 (rep 2)', 'PSMC4 (rep 1)']
rep_cycle_thresh_df = get_biospec_cols(biospec_df, rep_cycle_thresh_cols)
rep_cycle_avg_df, rep_cycle_thresh_avg_cols = avg_reps(rep_cycle_thresh_df)
rep_cycle_housekeeping_cols = ['GAPDH', 'PGK1']
rep_cycle_feature_cols = ['ALDH1A1', 'HSPA8', 'UBE2K', 'LAMB2', 'SKP1', 'PSMC4']
rep_cycle_avg_df = get_normalized_rna(rep_cycle_avg_df, rep_cycle_feature_cols, rep_cycle_housekeeping_cols)
if rep_cycle_avg_df is None:
    print('No RNA 2 to add to data')
else:
    print('Adding RNA 2')
    for col in rep_cycle_feature_cols:
        rep_cycle_avg_df[col] = rep_cycle_avg_df[col].astype(np.float64)
    collected_data = merge_dfs(collected_data, rep_cycle_avg_df[standard3_cols + rep_cycle_feature_cols])
    #collected_data = collected_data.merge(rep_cycle_avg_df[['PATNO','EVENT_ID'] + rep_cycle_feature_cols], \
    #                                      on=['PATNO','EVENT_ID'], how='outer', validate='one_to_one')
    shared_cols += rep_cycle_feature_cols
    rna_cols += rep_cycle_feature_cols

# divide count by size factor for that gene
# size factor = median across patients of gene count divided by geometric mean of counts for all housekeeping genes for that patient
count_cols = ['SRCAP', 'ZNF746', 'DHPR', 'SNCA-007', 'SNCA-3UTR-2', 'FBXO7-005', 'FBXO7-007', 'FBXO7-008', 'SNCA-3UTR-1', \
              'RPL13', 'FBXO7-010', 'DJ-1', 'GLT25D1', 'UBC', 'SNCA-E4E6', 'FBXO7-001', 'SNCA-E3E4', 'MON1B', 'GUSB']
count_df = get_biospec_cols(biospec_df, count_cols)
count_housekeeping_cols = ['SRCAP', 'RPL13', 'UBC', 'MON1B', 'GUSB']
count_df_feature_cols = ['DHPR', 'ZNF746', 'SNCA-007', 'SNCA-3UTR-2', 'FBXO7-005', 'FBXO7-007', 'FBXO7-008', 'SNCA-3UTR-1', \
                         'FBXO7-010', 'DJ-1', 'GLT25D1', 'SNCA-E4E6', 'FBXO7-001', 'SNCA-E3E4']
for col in count_housekeeping_cols:
    count_df[col] = count_df[col].astype(np.float64)
for col in count_df_feature_cols:
    count_df[col] = count_df[col].astype(np.float64)
count_df = count_df.dropna(subset=count_housekeeping_cols, how='all')
# nanmasked_count_housekeeping_arr = np.ma.masked_invalid(count_df[count_housekeeping_cols]) - gmean does this already
count_df['housekeeping_gmean'] = gmean(count_df[count_housekeeping_cols], axis=1)
norm_counts = count_df[standard3_cols]
for col in count_df_feature_cols:
    norm_counts[col] = count_df[col]/count_df['housekeeping_gmean']
count_size_facs = norm_counts[count_df_feature_cols].quantile(q=0.5, axis=0)
for col in count_df_feature_cols:
    count_df[col] = count_df[col]/count_size_facs[col]
print('Adding RNA 3')
for col in count_df_feature_cols:
    count_df[col] = count_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, count_df[standard3_cols + count_df_feature_cols])
#collected_data = collected_data.merge(count_df[['PATNO','EVENT_ID'] + count_df_feature_cols], \
#                                      on=['PATNO','EVENT_ID'], how='outer', validate='one_to_one')
shared_cols += count_df_feature_cols
rna_cols += count_df_feature_cols
shared_cols_dict['RNA'] = rna_cols

biochem_cols = ['4-Hydroxy-3-methoxymandelic acid', 'CSF Hemoglobin', '4-Hydroxy-3-methoxyphenylglycol (HMPG)', \
                'Noradrenaline (Norepinephrine)', '3,4-Dihydroxyphenylacetic acid (DOPAC)', 'Adrenaline (Epinephrine)', \
                'Homovanillic acid (HVA)', 'Metanephrine', 'Serotonin (5-HT)', 'Normetanephrine', '3-O-Methyldopamine', \
                '3,4-Dihydroxyphenylglycol (DOPEG)', 'Dopamine', '3,4-Dihydroxymandelic acid', \
                '3,4-Dihydroxyphenylalanine (DOPA)', 'Histamine', '3-Methoxytyrosine', \
                '5-Hydroxy-3-indoleacetic acid (5-HIAA)', 'C18 SM', 'C20 SM', 'C23 SM', 'C24:1 SM', 'C24 SM', 'Total SM', \
                'C22 SM', 'Probe activity assay', 'C16 SM']
biochem_df = get_biospec_cols(biospec_df, biochem_cols)
print('Adding biochem')
for col in biochem_cols:
    biochem_df[col] = biochem_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, biochem_df[standard3_cols + biochem_cols])
shared_cols += biochem_cols
shared_cols_dict['BIOCHEM'] = biochem_cols

# MoCA
moca_path = raw_datadir + 'Montreal_Cognitive_Assessment__MoCA_.csv'
moca_df = pd.read_csv(moca_path)
moca_symptom_cols = ['MCAALTTM', 'MCACUBE', 'MCACLCKC', 'MCACLCKN', 'MCACLCKH', 'MCALION', 'MCARHINO', 'MCACAMEL', 'MCAFDS', \
                     'MCABDS', 'MCAVIGIL', 'MCASER7', 'MCASNTNC', 'MCAVF', 'MCAABSTR', 'MCAREC1', 'MCAREC2', 'MCAREC3', \
                     'MCAREC4', 'MCAREC5', 'MCADATE', 'MCAMONTH', 'MCAYR', 'MCADAY', 'MCAPLACE', 'MCACITY']
for col in moca_symptom_cols:
    moca_df[col] = moca_df[col].astype(np.float64)
moca_df['unadjusted_moca'] = moca_df[moca_symptom_cols].sum(axis=1)
print('Adding MoCA')
collected_data = merge_dfs(collected_data, moca_df[standard3_cols + moca_symptom_cols + ['unadjusted_moca']])
collected_data['MOCA'] = np.where(np.logical_and(collected_data['EDUCYRS'] <= 12, collected_data['unadjusted_moca'] < 30), \
                                  collected_data['unadjusted_moca'] + 1, collected_data['unadjusted_moca'])
del collected_data['unadjusted_moca']
symptom_cols += moca_symptom_cols
total_cols.append('MOCA')
symptom_cols_dict['MOCA'] = moca_symptom_cols

# BJLO
bjlo_path = raw_datadir + 'Benton_Judgment_of_Line_Orientation.csv'
bjlo_df = pd.read_csv(bjlo_path)
bjlo_df['LAST_UPDATE'] = pd.to_datetime(bjlo_df['LAST_UPDATE'])
bjlo_df = bjlo_df.sort_values(by=standard3_cols + ['LAST_UPDATE'])
bjlo_df = bjlo_df.drop_duplicates(subset=standard3_cols, keep='last')
bjlo_symptom_cols = []
for idx in range(1,31):
    bjlo_symptom_cols.append('BJLOT'+str(idx))
for col in bjlo_symptom_cols:
    bjlo_df[col] = bjlo_df[col].astype(np.float64)
bjlo_df['BJLO'] = bjlo_df[bjlo_symptom_cols].sum(axis=1)
print('Adding BJLO')
collected_data = merge_dfs(collected_data, bjlo_df[standard3_cols + bjlo_symptom_cols + ['BJLO']])
symptom_cols += bjlo_symptom_cols
total_cols.append('BJLO')
symptom_cols_dict['BJLO'] = bjlo_symptom_cols

# Epworth sleepiness scale
ess_path = raw_datadir + 'Epworth_Sleepiness_Scale.csv'
ess_df = pd.read_csv(ess_path)
ess_symptom_cols = []
for idx in range(1,9):
    ess_symptom_cols.append('ESS'+str(idx))
for col in ess_symptom_cols:
    ess_df[col] = ess_df[col].astype(np.float64)
ess_df['ESS_sum'] = ess_df[ess_symptom_cols].sum(axis=1)
ess_df['EPWORTH'] = np.where(ess_df['ESS_sum'] < 10, 0, 1)
del ess_df['ESS_sum']
print('Adding ESS')
collected_data = merge_dfs(collected_data, ess_df[standard3_cols + ess_symptom_cols + ['EPWORTH']])
symptom_cols += ess_symptom_cols
total_cols.append('EPWORTH')
symptom_cols_dict['EPWORTH'] = ess_symptom_cols

# Geriatric depression scale
gds_path = raw_datadir + 'Geriatric_Depression_Scale__Short_.csv'
gds_df = pd.read_csv(gds_path)
# These columns need to be flipped so that higher is worse
gds_happy_cols = ['GDSSATIS', 'GDSGSPIR', 'GDSHAPPY', 'GDSALIVE', 'GDSENRGY']
gds_depressed_cols = ['GDSDROPD', 'GDSEMPTY', 'GDSBORED', 'GDSAFRAD', 'GDSHLPLS', 'GDSHOME', 'GDSMEMRY', 'GDSWRTLS', \
                      'GDSHOPLS', 'GDSBETER']
for col in gds_happy_cols:
    gds_df[col] = gds_df[col].astype(np.float64)
for col in gds_depressed_cols:
    gds_df[col] = gds_df[col].astype(np.float64)
gds_df['GDS_raw'] = 5 - gds_df[gds_happy_cols].sum(axis=1) + gds_df[gds_depressed_cols].sum(axis=1)
gds_df['GDSSHORT'] = np.where(gds_df['GDS_raw'] >= 5, 1, 0)
del gds_df['GDS_raw']
print('Adding GDS')
collected_data = merge_dfs(collected_data, gds_df[standard3_cols + gds_happy_cols + gds_depressed_cols + ['GDSSHORT']])
symptom_cols += gds_happy_cols
symptom_cols += gds_depressed_cols
total_cols.append('GDSSHORT')
symptom_cols_dict['GDSSHORT'] = gds_happy_cols + gds_depressed_cols

# Hopkins verbal learning test: 3 components: immediate recall, discrimination recognition, retention
hvlt_path = raw_datadir + 'Hopkins_Verbal_Learning_Test.csv'
hvlt_df = pd.read_csv(hvlt_path)
hvlt_df['HVLT_immed_recall'] = hvlt_df[['HVLTRT1','HVLTRT2','HVLTRT3']].sum(axis=1)
hvlt_df['HVLT_discrim_recog'] = hvlt_df['HVLTREC'] - hvlt_df[['HVLTFPRL','HVLTFPUN']].sum(axis=1)
hvlt_df['HVLT_retent'] = hvlt_df['HVLTRDLY']/hvlt_df[['HVLTRT2','HVLTRT3']].max(axis=1)
hvlt_symptom_cols = ['HVLTRT1', 'HVLTRT2', 'HVLTRT3', 'HVLTREC', 'HVLTFPRL', 'HVLTFPUN', 'HVLTRDLY']
hvlt_total_cols = ['HVLT_immed_recall', 'HVLT_discrim_recog', 'HVLT_retent']
print('Adding HVLT')
for col in hvlt_symptom_cols:
    hvlt_df[col] = hvlt_df[col].astype(np.float64)
for col in hvlt_total_cols:
    hvlt_df[col] = hvlt_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, hvlt_df[standard3_cols + hvlt_symptom_cols + hvlt_total_cols])
symptom_cols += hvlt_symptom_cols
total_cols += hvlt_total_cols
symptom_cols_dict['HVLT'] = hvlt_symptom_cols

# Letter-number sequencing
lns_path = raw_datadir + 'Letter_-_Number_Sequencing__PD_.csv'
lns_df = pd.read_csv(lns_path)
lns_symptom_cols = []
for i in range(1,8):
    lns_symptom_cols.append('LNS'+str(i)+'A')
    lns_symptom_cols.append('LNS'+str(i)+'B')
    lns_symptom_cols.append('LNS'+str(i)+'C')
lns_df[lns_symptom_cols].fillna(0, inplace=True)
for col in lns_symptom_cols:
    lns_df[col] = lns_df[col].astype(np.float64)
lns_df['LNS'] = lns_df[lns_symptom_cols].sum(axis=1)
print('Adding LNS')
collected_data = merge_dfs(collected_data, lns_df[standard3_cols + lns_symptom_cols + ['LNS']])
symptom_cols += lns_symptom_cols
total_cols.append('LNS')
symptom_cols_dict['LNS'] = lns_symptom_cols

# Compulsive-impulsive behavior questionnaire
quip_path = raw_datadir + 'QUIP_Current_Short.csv'
quip_df = pd.read_csv(quip_path)
quip_df['QUIP_A'] = np.clip(quip_df[['CNTRLGMB','TMGAMBLE']].sum(axis=1),0,1)
quip_df['QUIP_B'] = np.clip(quip_df[['CNTRLSEX', 'TMSEX']].sum(axis=1),0,1)
quip_df['QUIP_C'] = np.clip(quip_df[['CNTRLBUY', 'TMBUY']].sum(axis=1),0,1)
quip_df['QUIP_D'] = np.clip(quip_df[['CNTRLEAT', 'TMEAT']].sum(axis=1),0,1)
quip_df['QUIP_E'] = quip_df[['TMTORACT', 'TMTMTACT', 'TMTRWD']].sum(axis=1)
quip_df['QUIP'] = quip_df[['QUIP_A', 'QUIP_B', 'QUIP_C', 'QUIP_D', 'QUIP_E']].sum(axis=1)
quip_symptom_cols = ['CNTRLGMB','TMGAMBLE'] + ['CNTRLSEX', 'TMSEX'] + ['CNTRLBUY', 'TMBUY'] + ['CNTRLEAT', 'TMEAT'] + ['TMTORACT', 'TMTMTACT', 'TMTRWD'] + ['TMDISMED', 'CNTRLDSM']
print('Adding QUIP')
quip_df['TMDISMED'] = np.where(quip_df['TMDISMED'] == 'N', '0', '1')
quip_df['CNTRLDSM'] = np.where(quip_df['CNTRLDSM'] == 'N', '0', '1')
all_quip_cols = quip_symptom_cols + ['QUIP']
for col in all_quip_cols:
    quip_df[col] = quip_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, quip_df[standard3_cols + quip_symptom_cols + ['QUIP']])
symptom_cols += quip_symptom_cols
total_cols.append('QUIP')
symptom_cols_dict['QUIP'] = quip_symptom_cols

# autonomic dysfunction
scopa_aut_path = raw_datadir + 'SCOPA-AUT.csv'
scopa_aut_df = pd.read_csv(scopa_aut_path)
scopa_aut_df.fillna(0)
scopa_aut_df['SCAU_catheter'] = np.where(np.logical_or.reduce((scopa_aut_df['SCAU8']==9, \
                                                               scopa_aut_df['SCAU9']==9, \
                                                               scopa_aut_df['SCAU10']==9, \
                                                               scopa_aut_df['SCAU11']==9, \
                                                               scopa_aut_df['SCAU12']==9, \
                                                               scopa_aut_df['SCAU13']==9)), 1, 0)
catheter_cols = ['SCAU8','SCAU9','SCAU10','SCAU11','SCAU12','SCAU13']
for col in catheter_cols:
    scopa_aut_df[col] = np.where(scopa_aut_df[col]==9,3,scopa_aut_df[col])
scopa_aut_df['SCAU_sexNA'] = np.where(np.logical_or.reduce((scopa_aut_df['SCAU22']==9, \
                                                            scopa_aut_df['SCAU23']==9, \
                                                            scopa_aut_df['SCAU24']==9, \
                                                            scopa_aut_df['SCAU25']==9)), 1, 0)
sex_cols = ['SCAU22','SCAU23','SCAU24','SCAU25']
for col in sex_cols:
    scopa_aut_df[col] = np.where(scopa_aut_df[col]==9,0,scopa_aut_df[col])
scopa_aut_symptom_cols = []
for idx in range(1,26):
    scopa_aut_symptom_cols.append('SCAU'+str(idx))
scopa_aut_df['SCOPA-AUT'] = scopa_aut_df[scopa_aut_symptom_cols].sum(axis=1)
scopa_aut_symptom_cols += ['SCAU23A','SCAU26A','SCAU26B','SCAU26C','SCAU26D','SCAU_catheter','SCAU_sexNA']
print('Adding SCOPA-AUT')
all_scopa_aut_cols = scopa_aut_symptom_cols + ['SCOPA-AUT']
for col in all_scopa_aut_cols:
    scopa_aut_df[col] = scopa_aut_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, scopa_aut_df[standard3_cols + scopa_aut_symptom_cols + ['SCOPA-AUT']])
symptom_cols += scopa_aut_symptom_cols
total_cols.append('SCOPA-AUT')
symptom_cols_dict['SCOPA-AUT'] = scopa_aut_symptom_cols

# semantic fluency
semantic_fluency_path = raw_datadir + 'Semantic_Fluency.csv'
semantic_fluency_df = pd.read_csv(semantic_fluency_path)
semantic_fluency_symptom_cols = ['VLTANIM','VLTVEG','VLTFRUIT']
for col in semantic_fluency_symptom_cols:
    semantic_fluency_df[col] = semantic_fluency_df[col].astype(np.float64)
semantic_fluency_df['SEMANTIC_FLUENCY'] = semantic_fluency_df[semantic_fluency_symptom_cols].sum(axis=1)
print('Adding semantic fluency')
collected_data = merge_dfs(collected_data, semantic_fluency_df[standard3_cols + semantic_fluency_symptom_cols + ['SEMANTIC_FLUENCY']])
symptom_cols += semantic_fluency_symptom_cols
total_cols.append('SEMANTIC_FLUENCY')
symptom_cols_dict['SEMANTIC_FLUENCY'] = semantic_fluency_symptom_cols

# state-trait anxiety
# state: response to perceived threat vs. trait: day-to-day basis
state_trait_anxiety_path = raw_datadir + 'State-Trait_Anxiety_Inventory.csv'
state_trait_anxiety_df = pd.read_csv(state_trait_anxiety_path)
state_forward_qs = [3, 4, 6, 7, 9, 12, 13, 14, 17, 18]
state_forward_columns = []
state_backward_columns = []
for idx in range(1,21):
    if idx in state_forward_qs:
        state_forward_columns.append('STAIAD'+str(idx))
    else:
        state_backward_columns.append('STAIAD'+str(idx))
state_trait_anxiety_df['STATE_ANXIETY'] = state_trait_anxiety_df[state_forward_columns].sum(axis=1) \
    + 5*len(state_backward_columns) - state_trait_anxiety_df[state_backward_columns].sum(axis=1)
trait_forward_qs = [22, 24, 25, 28, 29, 31, 32, 35, 37, 38, 40]
trait_forward_columns = []
trait_backward_columns = []
for idx in range(21,41):
    if idx in trait_forward_qs:
        trait_forward_columns.append('STAIAD'+str(idx))
    else:
        trait_backward_columns.append('STAIAD'+str(idx))
state_trait_anxiety_df['TRAIT_ANXIETY'] = state_trait_anxiety_df[trait_forward_columns].sum(axis=1) \
    + 5*len(trait_backward_columns) - state_trait_anxiety_df[trait_backward_columns].sum(axis=1)
anxiety_total_cols = ['STATE_ANXIETY', 'TRAIT_ANXIETY']
print('Adding anxiety')
anxiety_symptom_cols = state_forward_columns + state_backward_columns + trait_forward_columns + trait_backward_columns
for col in anxiety_symptom_cols:
    state_trait_anxiety_df[col] = state_trait_anxiety_df[col].astype(np.float64)
for col in anxiety_total_cols:
    state_trait_anxiety_df[col] = state_trait_anxiety_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, state_trait_anxiety_df[standard3_cols + anxiety_symptom_cols + anxiety_total_cols])
symptom_cols += anxiety_symptom_cols
total_cols += anxiety_total_cols
symptom_cols_dict['ANXIETY'] = anxiety_symptom_cols

# UPSIT: UPenn Smell Identification Test
upsit_path = raw_datadir + 'University_of_Pennsylvania_Smell_ID_Test.csv'
upsit_df = pd.read_csv(upsit_path)
upsit_columns = []
for idx in range(1,5):
    upsit_columns.append('UPSITBK'+str(idx))
for col in upsit_columns:
    upsit_df[col] = upsit_df[col].astype(np.float64)
upsit_df['UPSIT'] = upsit_df[upsit_columns].sum(axis=1)
upsit_columns.append('UPSIT')
upsit_df['EVENT_ID'] = 'BL'
upsit_df['INFODT'] = pd.to_datetime(upsit_df['INFODT'])
upsit_df = upsit_df.sort_values(by=['INFODT'])
upsit_df = upsit_df.drop_duplicates(subset=['PATNO'], keep='first')
print('Adding UPSIT')
collected_data = merge_dfs(collected_data, upsit_df[standard3_cols + upsit_columns])
baseline_cols += upsit_columns
baseline_cols_dict['UPSIT'] = upsit_columns

# MDS-UPDRS I
mdsupdrs1_path1 = raw_datadir + 'MDS_UPDRS_Part_I.csv'
mdsupdrs1_path2 = raw_datadir + 'MDS_UPDRS_Part_I__Patient_Questionnaire.csv'
mdsupdrs1_df1 = pd.read_csv(mdsupdrs1_path1)
mdsupdrs1_df2 = pd.read_csv(mdsupdrs1_path2)
mdsupdrs1_df1 = mdsupdrs1_df1[standard3_cols + ['NP1COG','NP1HALL','NP1DPRS','NP1ANXS','NP1APAT','NP1DDS']]
mdsupdrs1_df2 = mdsupdrs1_df2[standard3_cols + ['NP1SLPN','NP1SLPD','NP1PAIN','NP1URIN','NP1CNST','NP1LTHD','NP1FATG']]
mdsupdrs1_df = mdsupdrs1_df1.merge(mdsupdrs1_df2, on=standard3_cols, how='inner', validate='one_to_one')
mdsupdrs1_symptom_cols = ['NP1COG', 'NP1HALL', 'NP1DPRS', 'NP1ANXS', 'NP1APAT', 'NP1DDS', 'NP1SLPN', 'NP1SLPD', 'NP1PAIN', \
                          'NP1URIN', 'NP1CNST', 'NP1LTHD', 'NP1FATG']
for col in mdsupdrs1_symptom_cols:
    mdsupdrs1_df[col] = mdsupdrs1_df[col].astype(np.float64)
mdsupdrs1_df['NUPDRS1'] = mdsupdrs1_df[mdsupdrs1_symptom_cols].sum(axis=1)
print('Adding MDS-UPDRS I')
collected_data = merge_dfs(collected_data, mdsupdrs1_df[standard3_cols + mdsupdrs1_symptom_cols + ['NUPDRS1']])
symptom_cols += mdsupdrs1_symptom_cols
total_cols.append('NUPDRS1')
symptom_cols_dict['NUPDRS1'] = mdsupdrs1_symptom_cols

# MDS-UPDRS IV
mdsupdrs4_path = raw_datadir + 'MDS_UPDRS_Part_IV.csv'
mdsupdrs4_df = pd.read_csv(mdsupdrs4_path)
mdsupdrs4_symptom_cols = ['NP4WDYSK','NP4DYSKI','NP4OFF','NP4FLCTI','NP4FLCTX','NP4DYSTN']
for col in mdsupdrs4_symptom_cols:
    mdsupdrs4_df[col] = mdsupdrs4_df[col].astype(np.float64)
mdsupdrs4_df['NUPDRS4'] = mdsupdrs4_df[mdsupdrs4_symptom_cols].sum(axis=1)
print('Adding MDS-UPDRS IV')
collected_data = merge_dfs(collected_data, mdsupdrs4_df[standard3_cols + mdsupdrs4_symptom_cols + ['NUPDRS4']])
symptom_cols += mdsupdrs4_symptom_cols
total_cols.append('NUPDRS4')
symptom_cols_dict['NUPDRS4'] = mdsupdrs4_symptom_cols

# Schwab and England assessment of daily living
schwab_england_path = raw_datadir + 'Modified_Schwab_+_England_ADL.csv'
schwab_england_df = pd.read_csv(schwab_england_path)
print('Adding MSEADLG')
schwab_england_df['MSEADLG'] = schwab_england_df['MSEADLG'].astype(np.float64)
collected_data = merge_dfs(collected_data, schwab_england_df[standard3_cols + ['MSEADLG']])
shared_cols.append('MSEADLG')
shared_cols_dict['MSEADLG'] = ['MSEADLG']
    
# Neurological exam
neuro_exam_path = raw_datadir + 'General_Neurological_Exam.csv'
neuro_exam_df = pd.read_csv(neuro_exam_path)
neuro_sp_cols = ['MSRARSP','MSLARSP','MSRLRSP','MSLLRSP','COFNRRSP','COFNLRSP','COHSRRSP','COHSLRSP','SENRARSP',\
                 'SENLARSP','SENRLRSP','SENLLRSP']
neuro_rfl_cols = ['RFLRARSP','RFLLARSP','RFLRLRSP','RFLLLRSP']
neuro_plr_cols = ['PLRRRSP','PLRLRSP']
neuro_exam_df = neuro_exam_df[['PATNO','EVENT_ID','INFODT']+neuro_sp_cols+neuro_rfl_cols+neuro_plr_cols]
neuro_exam_df[neuro_sp_cols] = np.where(neuro_exam_df[neuro_sp_cols]<2,1,0)
neuro_cols = neuro_sp_cols
for rfl_col in neuro_rfl_cols:
    # separate hypo and hyper, above 4 or empty mean not tested so mark as nan
    neuro_exam_df[rfl_col+'_hypo'] = np.where(neuro_exam_df[rfl_col]<2, 2-neuro_exam_df[rfl_col], 0)
    neuro_exam_df[rfl_col+'_hyper'] = np.where(neuro_exam_df[rfl_col]>2, neuro_exam_df[rfl_col]-2, 0)
    neuro_exam_df[rfl_col+'_hypo'] = np.where(neuro_exam_df[rfl_col]>4, np.nan, neuro_exam_df[rfl_col+'_hypo'])
    neuro_exam_df[rfl_col+'_hyper'] = np.where(neuro_exam_df[rfl_col]>4, np.nan, neuro_exam_df[rfl_col+'_hyper'])
    neuro_exam_df[rfl_col+'_hypo'] = np.where(pd.isnull(neuro_exam_df[rfl_col]), np.nan, neuro_exam_df[rfl_col+'_hypo'])
    neuro_exam_df[rfl_col+'_hyper'] = np.where(pd.isnull(neuro_exam_df[rfl_col]), np.nan, neuro_exam_df[rfl_col+'_hyper'])
    del neuro_exam_df[rfl_col]
    neuro_cols += [rfl_col+'_hypo', rfl_col+'_hyper']
for plr_col in neuro_plr_cols:
    # separate indeterminate from untested and test value
    neuro_exam_df[plr_col+'_indet'] = np.where(neuro_exam_df[plr_col]==2, 1, 0)
    neuro_exam_df[plr_col+'_indet'] = np.where(neuro_exam_df[plr_col]>2, np.nan, neuro_exam_df[plr_col+'_indet'])
    neuro_exam_df[plr_col] = np.where(neuro_exam_df[plr_col]<2, neuro_exam_df[plr_col], np.nan)
    neuro_cols += [plr_col, plr_col+'_indet']
cranial_exam_path = raw_datadir + 'Neurological_Exam_-_Cranial_Nerves.csv'
cranial_exam_df = pd.read_csv(cranial_exam_path)
cranial_sp_cols = ['CN1RSP','CN2RSP','CN346RSP','CN5RSP','CN7RSP','CN8RSP','CN910RSP','CN11RSP','CN12RSP']
neuro_cols += cranial_sp_cols
cranial_exam_df = cranial_exam_df[standard3_cols+cranial_sp_cols]
# above 2 means untested
cranial_exam_df[cranial_sp_cols] = np.where(cranial_exam_df[cranial_sp_cols]<2, cranial_exam_df[cranial_sp_cols], np.nan)
neuro_combined_df = neuro_exam_df.merge(cranial_exam_df, how='outer', validate='one_to_one')
print('Adding neuro')
for col in neuro_cols:
    neuro_combined_df[col] = neuro_combined_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, neuro_combined_df[standard3_cols + neuro_cols])
shared_cols += neuro_cols
shared_cols_dict['NEURO_EXAM'] = neuro_cols

# PASE leisure activities
pase_leisure_path = raw_datadir + 'PASE_-_Leisure_Time_Activity.csv'
pase_leisure_df = pd.read_csv(pase_leisure_path)
pase_leisure_cols = []
pase_leisure_restruc_df = pase_leisure_df[['PATNO','EVENT_ID']].drop_duplicates()
for i in range(1,7):
    pase_leisure_col = 'PASE_LEIS' + str(i)
    pase_leisure_cols.append(pase_leisure_col)
    pase_leisure_q_df = pase_leisure_df.loc[pase_leisure_df['QUESTNO'] == i]
    pase_leisure_q_df.rename(columns={'ACTVOFT': pase_leisure_col}, inplace=True)
    pase_leisure_restruc_df = pase_leisure_restruc_df.merge(pase_leisure_q_df[['PATNO','EVENT_ID', pase_leisure_col]], how = 'left', validate = 'one_to_one')
for col in pase_leisure_cols:
    pase_leisure_restruc_df[col] = pase_leisure_restruc_df[col].astype(np.float64)
collected_data = collected_data.merge(pase_leisure_restruc_df, on=['PATNO','EVENT_ID'], how='outer', validate='one_to_one')
shared_cols += pase_leisure_cols
eventid_only_cols += pase_leisure_cols
shared_cols_dict['PASE_LEISURE'] = pase_leisure_cols

# PASE household activities
pase_household_path = raw_datadir + 'PASE_-_Household_Activity.csv'
pase_household_df = pd.read_csv(pase_household_path)
pase_household_cols = ['LTHSWRK', 'HVYHSWRK', 'HMREPR', 'LAWNWRK', 'OUTGARDN', 'CAREGVR', 'WRKVL', 'WRKVLHR', 'WRKVLACT']
for col in pase_household_cols[:-2]:
    pase_household_df[col]  = pase_household_df[col] - 1
pase_household_df[['WRKVLHR','WRKVLACT']].fillna(0, inplace = True)
for col in pase_household_cols:
    pase_household_df[col] = pase_household_df[col].astype(np.float64)
collected_data = collected_data.merge(pase_household_df[['PATNO','EVENT_ID'] + pase_household_cols], on=['PATNO','EVENT_ID'], \
                                      how='outer', validate='one_to_one')
shared_cols += pase_household_cols
eventid_only_cols += pase_household_cols
shared_cols_dict['PASE_HOUSEHOLD'] = pase_household_cols

# Symbol-digit modalities
symb_dig_path = raw_datadir + 'Symbol_Digit_Modalities.csv'
symb_dig_df = pd.read_csv(symb_dig_path)
symb_dig_cols = ['SDMTOTAL', 'DVSD_SDM', 'DVT_SDM']
print('Adding symb-dig')
for col in symb_dig_cols:
    symb_dig_df[col] = symb_dig_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, symb_dig_df[standard3_cols + symb_dig_cols])
shared_cols += symb_dig_cols
shared_cols_dict['SYMB_DIG'] = symb_dig_cols

# Vital signs
vital_signs_path = raw_datadir + 'Vital_Signs.csv'
vital_signs_df = pd.read_csv(vital_signs_path)
vital_signs_cols = ['WGTKG', 'HTCM', 'TEMPC', 'BPARM', 'SYSSUP', 'DIASUP', 'HRSUP', 'SYSSTND', 'DIASTND', 'HRSTND']
print('Adding vital signs')
for col in vital_signs_cols:
    vital_signs_df[col] = vital_signs_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, vital_signs_df[standard3_cols + vital_signs_cols])
shared_cols += vital_signs_cols
shared_cols_dict['VITALS'] = vital_signs_cols

# Blood chemistry hematology
# RBC Morphology has word entries
# 'LSILORNG', 'LSIHIRNG', 'LUSLORNG', 'LUSHIRNG' are reference ranges, so they are not stored
# LRESFLG: H, L, HT, LT, HP, LP flags for high/low in increasing level of severity -> turn into binary indicators
# drop if test value is 'Not ident. w/i'
blood_hema_path = raw_datadir + 'Blood_Chemistry___Hematology.csv'
blood_hema_df = pd.read_csv(blood_hema_path)
print('Adding blood hema')
blood_hema_df = blood_hema_df.loc[pd.isnull(blood_hema_df['LTSTCOMM'])]
blood_hema_df = blood_hema_df.loc[blood_hema_df['EVENT_ID']!='RETEST']
blood_hema_df['LCOLLDT'] = pd.to_datetime(blood_hema_df['LCOLLDT'])
blood_hema_df['COLLTM'] = pd.to_datetime(blood_hema_df['COLLTM'])
blood_hema_df.rename(columns={'LCOLLDT':'INFODT'}, inplace=True)
blood_hema_df = blood_hema_df.loc[np.logical_and(blood_hema_df['LSIRES']!='Not ident. w/i', \
                                                 blood_hema_df['LUSRES']!='Not ident. w/i')]
blood_hema_restruc_df = blood_hema_df[standard3_cols].drop_duplicates()
feat_counts = blood_hema_df.LTSTNAME.value_counts()
blood_hema_cols = []
orig_blood_hema_cols = ['LSIRES', 'LUSRES', 'LRESFLG']
lresflgs = ['H', 'L', 'HT', 'LT', 'HP', 'LP']
for feat in feat_counts.keys():
    if feat_counts[feat] < 700:
        continue
    blood_hema_feat_df = blood_hema_df.loc[blood_hema_df['LTSTNAME']==feat]
    blood_hema_feat_df = blood_hema_feat_df.sort_values(by=['COLLTM'])
    blood_hema_feat_df = blood_hema_feat_df[standard3_cols + orig_blood_hema_cols].drop_duplicates(subset=standard3_cols, keep='last')
    if feat == 'RBC Morphology':
        feat_col_names = []
        lsires_feat_vals = blood_hema_feat_df.LSIRES.value_counts().keys() # unique() might contain nan vals
        for val in lsires_feat_vals:
            feat_val_col_name = feat + '_LSIRES:' + val
            blood_hema_feat_df[feat_val_col_name] = np.where(blood_hema_feat_df['LSIRES']==val, 1, 0)
            feat_col_names.append(feat_val_col_name)
        lusres_feat_vals = blood_hema_feat_df.LUSRES.value_counts().keys()
        for val in lusres_feat_vals:
            feat_val_col_name = feat + '_LUSRES:' + val
            blood_hema_feat_df[feat_val_col_name] = np.where(blood_hema_feat_df['LUSRES']==val, 1, 0)
            feat_col_names.append(feat_val_col_name)
    else:
        blood_hema_feat_df.rename(columns={'LSIRES': feat + '_LSIRES', 'LUSRES': feat + '_LUSRES'}, inplace=True)
        feat_col_names = [feat + '_LSIRES', feat + '_LUSRES']
    for flag in lresflgs:
        if len(blood_hema_feat_df.loc[blood_hema_feat_df['LRESFLG']==flag]) > 0:
            feat_flag = feat + '_LRESFLG:' + flag
            blood_hema_feat_df[feat_flag] = np.where(blood_hema_feat_df['LRESFLG']==flag, 1, 0)
            feat_col_names.append(feat_flag)
    blood_hema_restruc_df = blood_hema_restruc_df.merge(blood_hema_feat_df[standard3_cols + feat_col_names], how = 'left', validate = 'one_to_one')
    blood_hema_cols += feat_col_names
for col in blood_hema_cols:
    blood_hema_restruc_df[col] = blood_hema_restruc_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, blood_hema_restruc_df[standard3_cols + blood_hema_cols])
shared_cols += blood_hema_cols
shared_cols_dict['BLOOD_HEMA'] = blood_hema_cols

# Cognitive categorization
# record COGDECLN, FNCDTCOG, and COGSTATE where confidence is high (COGDXCL = 1 or 2)
cog_cat_path = raw_datadir + 'Cognitive_Categorization.csv'
cog_cat_df = pd.read_csv(cog_cat_path)
cog_cat_df = cog_cat_df.loc[cog_cat_df['COGDXCL'] <= 2]
cog_cat_cols = ['COGDECLN', 'FNCDTCOG', 'COGSTATE']
print('Adding cog cat')
for col in cog_cat_cols:
    cog_cat_df[col] = cog_cat_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, cog_cat_df[standard3_cols + cog_cat_cols])
shared_cols += cog_cat_cols
shared_cols_dict['COG_CAT'] = cog_cat_cols

# DTI imaging
# TODO: may need to derive feature, e.g. ratio or difference between ROI and REF
dti_path = raw_datadir + 'DTI_Regions_of_Interest.csv'
dti_df = pd.read_csv(dti_path)
dti_cols = []
orig_dti_cols = ['ROI1', 'ROI2', 'ROI3', 'ROI4', 'ROI5', 'ROI6', 'REF1', 'REF2']
dti_df['RUNDATE'] = pd.to_datetime(pd.to_datetime(dti_df['RUNDATE']).dt.strftime('%m/%Y'))
dti_df['INFODT'] = pd.to_datetime(dti_df['INFODT'])
dti_restruc_df = dti_df[['PATNO','RUNDATE','INFODT']].drop_duplicates(subset=['PATNO','RUNDATE'])
for measure in dti_df.Measure.unique():
    dti_measure_df = dti_df.loc[dti_df['Measure']==measure]
    dti_measure_mean_df = dti_measure_df.groupby(by=['PATNO','RUNDATE']).agg({'ROI1': {measure + '_ROI1_mean':'mean'}, \
                                                                              'ROI2': {measure + '_ROI2_mean': 'mean'}, \
                                                                              'ROI3': {measure + '_ROI3_mean': 'mean'}, \
                                                                              'ROI4': {measure + '_ROI4_mean': 'mean'}, \
                                                                              'ROI5': {measure + '_ROI5_mean':'mean'}, \
                                                                              'ROI6': {measure + '_ROI6_mean': 'mean'}, \
                                                                              'REF1': {measure + '_REF1_mean': 'mean'}, \
                                                                              'REF2': {measure + '_REF2_mean': 'mean'}})
    dti_measure_mean_df.columns = dti_measure_mean_df.columns.droplevel(0)
    dti_measure_mean_df = dti_measure_mean_df.reset_index()
    dti_col_measure_dict = dict()
    dti_col_measure_names = []
    for col in orig_dti_cols:
        col_measure_name = measure + '_' + col + '_mean'
        dti_col_measure_names.append(col_measure_name)
    dti_restruc_df = dti_restruc_df.merge(dti_measure_mean_df[['PATNO', 'RUNDATE'] + dti_col_measure_names], \
                                          on = ['PATNO', 'RUNDATE'], how = 'left', validate = 'one_to_one')
    dti_cols += dti_col_measure_names
for col in dti_cols:
    dti_restruc_df[col] = dti_restruc_df[col].astype(np.float64)
# exact merge on INFODT catches 259 of 263 of the DTI imaging results and makes the most sense
collected_data = collected_data.merge(dti_restruc_df[['PATNO','INFODT'] + dti_cols], on=['PATNO','INFODT'], how='outer', validate='many_to_one')
shared_cols += dti_cols
infodt_only_cols += dti_cols
shared_cols_dict['DTI'] = dti_cols

# General medical history
# MHACTRES = 2 and MHHX = 1 indicates currently active co-morbidity
med_hist_path = raw_datadir + 'General_Medical_History.csv'
med_hist_df = pd.read_csv(med_hist_path)
med_hist_df = med_hist_df.loc[np.logical_and(med_hist_df['MHACTRES']==2, med_hist_df['MHHX'] == 1)]
med_hist_df['MHCAT'] = med_hist_df['MHCAT'].str.lower()
med_hist_restruc_df = med_hist_df[standard3_cols].drop_duplicates()
med_hist_cats = med_hist_df.MHCAT.unique().tolist()
med_hist_cols = []
for question in med_hist_cats:
    med_hist_question_df = med_hist_df.loc[med_hist_df['MHCAT']==question][standard3_cols].drop_duplicates()
    med_hist_q_col = 'MedHist_' + question
    med_hist_cols.append(med_hist_q_col)
    med_hist_question_df[med_hist_q_col] = 1
    med_hist_restruc_df = med_hist_restruc_df.merge(med_hist_question_df, how = 'left', validate = 'one_to_one')
med_hist_restruc_df.fillna(0, inplace=True)
print('Adding med hist')
for col in med_hist_cols:
    med_hist_restruc_df[col] = med_hist_restruc_df[col].astype(np.float64)
med_hist_restruc_df['EVENT_ID'] = 'SC'
collected_data = merge_dfs(collected_data, med_hist_restruc_df)
screening_cols += med_hist_cols
screening_cols_dict['MEDHIST'] = med_hist_cols

# General physical exam
# Although 4 patients were given a 2nd exam, rest at screening only, so treat as a baseline feature
phys_exam_path = raw_datadir + 'General_Physical_Exam.csv'
phys_exam_df = pd.read_csv(phys_exam_path)
phys_exam_df = phys_exam_df.loc[phys_exam_df['EVENT_ID']=='SC']
phys_exam_df['ABNORM'] = np.where(phys_exam_df['ABNORM']>=2, 0, phys_exam_df['ABNORM'])
phys_exam_cats = phys_exam_df.PECAT.unique().tolist()
phys_exam_restruc_df = phys_exam_df[standard3_cols].drop_duplicates()
phys_exam_cols = []
for col in phys_exam_cats:
    phys_exam_col_df = phys_exam_df.loc[phys_exam_df['PECAT']==col][standard3_cols + ['ABNORM']]
    phys_exam_col_df = phys_exam_col_df.sort_values(by=standard3_cols + ['ABNORM'])
    phys_exam_col_df = phys_exam_col_df.drop_duplicates(subset=standard3_cols, keep='last')
    phys_exam_col_name = 'PhysExam_' + col
    phys_exam_col_df.rename(columns={'ABNORM': phys_exam_col_name}, inplace=True)
    phys_exam_restruc_df = phys_exam_restruc_df.merge(phys_exam_col_df, how = 'left', validate = 'one_to_one')
    phys_exam_cols.append(phys_exam_col_name)
print('Adding phys exam')
for col in phys_exam_cols:
    #print(col)
    phys_exam_restruc_df[col] = phys_exam_restruc_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, phys_exam_restruc_df)
screening_cols += phys_exam_cols
screening_cols_dict['PHYS_EXAM'] = phys_exam_cols

# Diagnostic features
diag_feat_path = raw_datadir + 'Diagnostic_Features.csv'
diag_feat_df = pd.read_csv(diag_feat_path)
diag_feat_cols = ['DFSTROKE', 'DFRSKFCT', 'DFPRESNT', 'DFRPROG', 'DFSTATIC', 'DFHEMPRK', 'DFAGESX', 'DFOTHCRS', 'DFRTREMP', 'DFRTREMA', 'DFPATREM', 'DFOTHTRM', 'DFRIGIDP', 'DFRIGIDA', 'DFAXRIG', 'DFUNIRIG', 'DFTONE', 'DFOTHRIG', 'DFBRADYP', 'DFBRADYA', 'DFAKINES', 'DFBRPLUS', 'DFOTHABR', 'DFPGDIST', 'DFGAIT', 'DFFREEZ', 'DFFALLS', 'DFOTHPG', 'DFPSYCH', 'DFCOGNIT', 'DFDYSTON', 'DFCHOREA', 'DFMYOCLO', 'DFOTHHYP', 'DFHEMTRO', 'DFPSHYPO', 'DFSEXDYS', 'DFURDYS', 'DFBWLDYS', 'DFOCULO', 'DFEYELID', 'DFNEURAB', 'DFDOPRSP', 'DFRAPSPE', 'DFBULBAR', 'DFCTSCAN', 'DFMRI', 'DFATYP']
print('Adding diag feat')
# N means medication/imaging not given -> 0
diag_feat_df['DFDOPRSP'] = np.where(diag_feat_df['DFDOPRSP']=='N', 0, diag_feat_df['DFDOPRSP'])
diag_feat_df['DFCTSCAN'] = np.where(diag_feat_df['DFCTSCAN']=='N', 0, diag_feat_df['DFCTSCAN'])
diag_feat_df['DFMRI'] = np.where(diag_feat_df['DFMRI']=='N', 0, diag_feat_df['DFMRI'])
for col in diag_feat_cols:
    diag_feat_df[col] = diag_feat_df[col].astype(np.float64)
collected_data = merge_dfs(collected_data, diag_feat_df[standard3_cols + diag_feat_cols])
shared_cols += diag_feat_cols
shared_cols_dict['DIAG_FEATS'] = diag_feat_cols

# fill in BIRTHDT and PDDXDT to all places that match PATNO
birthdt_df = collected_data[['PATNO','BIRTHDT']].dropna().drop_duplicates()
del collected_data['BIRTHDT']
collected_data = collected_data.merge(birthdt_df, on=['PATNO'], how='left', validate='many_to_one')
pddxdt_df = collected_data[['PATNO','PDDXDT']].dropna().drop_duplicates()
del collected_data['PDDXDT']
collected_data = collected_data.merge(pddxdt_df, on=['PATNO'], how='left', validate='many_to_one')
# use prodromal questionnaire for prodromal diagnostic date
prodromal_diag_phenoconv_df = collected_data[['PATNO','PRODROMAL_DIAG:PHENOCONV','INFODT']].dropna(subset=['PRODROMAL_DIAG:PHENOCONV'])
prodromal_diag_phenoconv_df = prodromal_diag_phenoconv_df.loc[prodromal_diag_phenoconv_df['PRODROMAL_DIAG:PHENOCONV']==1]
prodromal_diag_phenoconv_df['INFODT'] = pd.to_datetime(prodromal_diag_phenoconv_df['INFODT'])
prodromal_diag_phenoconv_df = prodromal_diag_phenoconv_df.sort_values(by=['INFODT'])
prodromal_diag_phenoconv_df = prodromal_diag_phenoconv_df.drop_duplicates(subset=['PATNO'], keep='first')
prodromal_diag_phenoconv_df.rename(columns={'INFODT':'PRODROMA_DXDT'}, inplace=True)
collected_data = collected_data.merge(prodromal_diag_phenoconv_df[['PATNO','PRODROMA_DXDT']], on=['PATNO'], how='left', \
                                      validate='many_to_one')
collected_data['PDDXDT'] = np.where(collected_data['PATNO'].isin(set(prodromal_diag_phenoconv_df.PATNO.unique())), \
                                    collected_data['PRODROMA_DXDT'], collected_data['PDDXDT'])
del collected_data['PRODROMA_DXDT']

patient_status_path = raw_datadir + 'Patient_Status.csv'
patient_status_df = pd.read_csv(patient_status_path)
collected_data = collected_data.merge(patient_status_df[['PATNO','ENROLL_DATE']], on=['PATNO'], how='left', validate='many_to_one')

# calculate age, disease duration, and time since baseline visit at each visit
collected_data['AGE'] = (pd.to_datetime(collected_data['INFODT']) - pd.to_datetime(collected_data['BIRTHDT']))/ np.timedelta64(1, 'Y')
del collected_data['BIRTHDT']
shared_cols.append('AGE')
shared_cols_dict['AGE'] = ['AGE']
collected_data['INFODT_DIS_DUR'] = (pd.to_datetime(collected_data['INFODT']) \
                                    - pd.to_datetime(collected_data['PDDXDT']))/ np.timedelta64(1, 'Y')
collected_data['INFODT_TIME_SINCE_ENROLL'] = (pd.to_datetime(collected_data['INFODT']) - pd.to_datetime(collected_data['ENROLL_DATE']))/ np.timedelta64(1, 'Y')
collected_data['INFODT_TIME_SINCE_ENROLL'] = collected_data['INFODT_TIME_SINCE_ENROLL'].astype(np.float64)
del collected_data['ENROLL_DATE']

# convert ST to its appropriate visit
st_catalog_path = raw_datadir + 'ST_CATALOG.csv'
st_catalog_df = pd.read_csv(st_catalog_path)
collected_data = collected_data.merge(st_catalog_df[['PATNO','STRPLCVS']], on=['PATNO'], how='left', validate='many_to_one')
collected_data['EVENT_ID'] = np.where(np.logical_and(collected_data['EVENT_ID']=='ST', \
                                                     ~pd.isnull(collected_data['STRPLCVS'])), \
                                      collected_data['STRPLCVS'], collected_data['EVENT_ID'])
del collected_data['STRPLCVS']
# Drop unscheduled visits starting with U since little data taken at unscheduled visits anyways
collected_data = collected_data.loc[~collected_data['EVENT_ID'].str.startswith('U', na=False)]

event_id_dur_dict = {'SC': 0, 'BL': 1.5, 'V01': 4.5, 'V02': 7.5, 'V03': 10.5, 'V04': 13.5, 'V05': 19.5, 'V06': 25.5, \
                     'V07': 31.5, 'V08': 37.5, 'V09': 43.5, 'V10': 49.5, 'V11': 55.5, 'V12': 61.5, 'V13': 73.5, 'V14': 85.5, \
                     'V15': 97.5, 'BSL': 1.5, 'PV02': 7.5, 'PV04': 13.5, 'PV05': 19.5, 'PV06': 25.5, \
                     'PV07': 31.5, 'PV08': 37.5, 'PV09': 43.5, 'PV10': 49.5, 'PV11': 55.5, 'PV12': 61.5, \
                     'P13': 79.5, 'P14': 91.5, 'P15': 103.5, 'V16': 109.5, 'P16': 115.5, 'V17': 121.5, 'P17': 127.5, \
                     'V18': 133.5, 'P18': 139.5, 'V19': 145.5, 'P19': 151.5, 'V20': 157.5}
collected_data['EVENT_ID_DUR'] = collected_data['INFODT_TIME_SINCE_ENROLL']
for event_id in event_id_dur_dict:
    collected_data['EVENT_ID_DUR'] = np.where(collected_data['EVENT_ID']==event_id, event_id_dur_dict[event_id]/12., \
                                              collected_data['EVENT_ID_DUR'])
collected_data['EVENT_ID_DUR'] = collected_data['EVENT_ID_DUR'].astype(np.float64)
#collected_data['DT_FROM_CONSENTDT_EVENT_ID'] = collected_data['CONSNTDT'] \
#    + pd.TimedeltaIndex(collected_data['EVENT_ID_DUR'], 'Y') # doesn't work
#collected_data['DT_FROM_CONSENTDT_EVENT_ID'] = pd.to_datetime(collected_data['DT_FROM_CONSENTDT_EVENT_ID'])
collected_data['DIS_DUR_BY_CONSENTDT'] = (pd.to_datetime(collected_data['CONSNTDT']) \
                                          - pd.to_datetime(collected_data['PDDXDT']))/ np.timedelta64(1, 'Y') \
                                        + collected_data['EVENT_ID_DUR']
del collected_data['CONSNTDT']
del collected_data['PDDXDT']
shared_cols.remove('PDDXDT')
time_feats = ['INFODT_DIS_DUR', 'INFODT_TIME_SINCE_ENROLL', 'EVENT_ID_DUR', 'DIS_DUR_BY_CONSENTDT']
screening_cols += time_feats
screening_cols_dict['TIME_FEATS'] = time_feats
baseline_cols += time_feats
baseline_cols_dict['TIME_FEATS'] = time_feats
total_cols += time_feats
symptom_cols += time_feats
symptom_cols_dict['TIME_FEATS'] = time_feats
shared_cols += time_feats
shared_cols_dict['TIME_FEATS'] = time_feats

binary_cols = set()
all_cols = screening_cols + baseline_cols + symptom_cols + total_cols + shared_cols
for col in all_cols:
    col_value_counts = collected_data[[col]].dropna()[col].value_counts()
    if len(col_value_counts) <= 2 and set(col_value_counts.keys()).issubset(set({0,1})):
        binary_cols.add(col)

print('outputting')
output_dir = ppmi_dir + 'pipeline_output_asof_' + download_date + '_using_CMEDTM/'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
with open(output_dir + 'binary_cols.pkl', 'w') as f:
    pickle.dump(binary_cols, f)
with open(output_dir + 'screening_cols.pkl', 'w') as f:
    pickle.dump(screening_cols_dict, f)
with open(output_dir + 'baseline_cols.pkl', 'w') as f:
    pickle.dump(baseline_cols_dict, f)
with open(output_dir + 'questions_cols.pkl', 'w') as f:
    pickle.dump(symptom_cols_dict, f)
with open(output_dir + 'other_cols.pkl', 'w') as f:
    pickle.dump(shared_cols_dict, f)
    
def output_baseline_features(df, cols):
    output_str = ''
    agg_stats = []
    agg_stat_names = []
    for feat in cols:
        if feat == 'CNO':
            num_cnos = df.CNO.nunique()
            agg_stats.append(num_cnos)
            agg_stat_names.append('CNO num sites')
            continue
        nonnan_feat_df = df.loc[~pd.isnull(df[feat])][['PATNO',feat]]
        if feat in binary_cols:
            if len(nonnan_feat_df) == 0:
                binary_freq1 = 0.
                feat_num_patnos = 0
            else:
                feat_num_patnos = nonnan_feat_df.PATNO.nunique()
                feat_vals = nonnan_feat_df[feat].value_counts()
                if 1 in feat_vals.keys():
                    binary_freq1 = feat_vals[1]/float(len(nonnan_feat_df))
                else:
                    binary_freq1 = 0.
            output_str += feat + ': ' + str(binary_freq1) + ', ' + str(feat_num_patnos) + '\n'
            agg_stats += [binary_freq1, feat_num_patnos]
            agg_stat_names += [feat + '_freq', feat + '_num_patnos']
        else:
            if len(nonnan_feat_df) == 0:
                feat_10 = 0.
                feat_mean = 0.
                feat_90 = 0.
                feat_num_patnos = 0.
            else:
                feat_num_patnos = nonnan_feat_df.PATNO.nunique()
                feat_arr = nonnan_feat_df[feat].values.astype(np.float64)
                nonnan_feat_df_noinf = nonnan_feat_df.loc[nonnan_feat_df[feat]!=float('+inf')]
                nonnan_feat_df_noinf = nonnan_feat_df_noinf.loc[nonnan_feat_df_noinf[feat]!=float('-inf')]
                feat_arr_noinf = nonnan_feat_df_noinf[feat].values.astype(np.float64)
                feat_10 = np.percentile(feat_arr, 10)
                feat_mean = np.mean(feat_arr_noinf)
                feat_90 = np.percentile(feat_arr, 90)
            output_str += feat + ': ' + str(feat_10) + ', ' + str(feat_mean) + ', ' + str(feat_90) + ', ' \
                + str(feat_num_patnos) + '\n'
            agg_stats += [feat_10, feat_mean, feat_90, feat_num_patnos]
            agg_stat_names += [feat + '_10', feat + '_mean', feat + '_90', feat + '_num_patnos']
    return output_str, agg_stats, agg_stat_names

def output_changing_features(df, cols):
    output_str = ''
    agg_stats = []
    agg_stat_names = []
    for feat in cols:
        nonnan_feat_df = df.loc[~pd.isnull(df[feat])]
        if feat in binary_cols:
            if len(nonnan_feat_df) == 0:
                binary_freq1 = 0.
                feat_num_patnos = 0
                avg_num_visits = 0.
            else:
                feat_num_patnos = nonnan_feat_df.PATNO.nunique()
                avg_num_visits = len(nonnan_feat_df)/float(feat_num_patnos)
                feat_vals = nonnan_feat_df[feat].value_counts()
                if 1 in feat_vals.keys():
                    binary_freq1 = feat_vals[1]/float(len(nonnan_feat_df))
                else:
                    binary_freq1 = 0
            output_str += feat + ': ' + str(binary_freq1) + ', ' + str(feat_num_patnos) + ', ' + str(avg_num_visits) + '\n'
            agg_stats += [binary_freq1, feat_num_patnos, avg_num_visits]
            agg_stat_names += [feat + '_freq', feat + '_num_patnos', feat + '_avg_num_visits']
        else:
            if len(nonnan_feat_df) == 0:
                feat_10 = 0.
                feat_mean = 0.
                feat_90 = 0.
                feat_num_patnos = 0
                avg_num_visits = 0.
            else:
                feat_num_patnos = nonnan_feat_df.PATNO.nunique()
                avg_num_visits = len(nonnan_feat_df)/float(feat_num_patnos)
                feat_vals = nonnan_feat_df[feat].value_counts()
                feat_arr = nonnan_feat_df[feat].values.astype(np.float64)
                nonnan_feat_df_noinf = nonnan_feat_df.loc[nonnan_feat_df[feat]!=float('+inf')]
                nonnan_feat_df_noinf = nonnan_feat_df_noinf.loc[nonnan_feat_df_noinf[feat]!=float('-inf')]
                feat_arr_noinf = nonnan_feat_df_noinf[feat].values.astype(np.float64)
                feat_10 = np.percentile(feat_arr, 10)
                feat_mean = np.mean(feat_arr_noinf)
                feat_90 = np.percentile(feat_arr, 90)
            output_str += feat + ': ' + str(feat_10) + ', ' + str(feat_mean) + ', ' + str(feat_90) + ', ' \
                + str(feat_num_patnos) + ', ' + str(avg_num_visits) + '\n'
            agg_stats += [feat_10, feat_mean, feat_90, feat_num_patnos, avg_num_visits]
            agg_stat_names += [feat + '_10', feat + '_mean', feat + '_90', feat + '_num_patnos', feat + '_avg_num_visits']
    return output_str, agg_stats, agg_stat_names

def output_df(df, cohort_name):
    # write 4 csvs per dataframe: 3 across time (symptoms, totals, shared), 1 for baseline
    # summary stats for dataframe to csv
    symptom_filename = output_dir + cohort_name + '_questions_across_time.csv'
    symptom_df = df[standard3_cols + symptom_cols]
    symptom_df.to_csv(symptom_filename, index=False)
    total_filename = output_dir + cohort_name + '_totals_across_time.csv'
    total_df = df[standard3_cols + total_cols]
    total_df.to_csv(total_filename, index=False)
    shared_filename = output_dir + cohort_name + '_other_across_time.csv'
    shared_df = df[standard3_cols + shared_cols]
    shared_df.to_csv(shared_filename, index=False)
    screening_filename = output_dir + cohort_name + '_screening.csv'
    baseline_filename = output_dir + cohort_name + '_baseline.csv'
    sc_baseline_df = df.loc[df['EVENT_ID']=='SC'][standard3_cols + screening_cols]
    assert len(sc_baseline_df) == sc_baseline_df.PATNO.nunique()
    sc_baseline_df.to_csv(screening_filename, index=False)
    bl_baseline_df = df.loc[df['EVENT_ID']=='BL'][standard3_cols + baseline_cols]
    bl_missing_patnos = set(df.PATNO.unique()).difference(set(bl_baseline_df.PATNO.unique()))
    bl_missing_patnos_df = df.loc[df['PATNO'].isin(bl_missing_patnos)][standard3_cols + baseline_cols].dropna(how='all')
    bl_missing_patnos_df = bl_missing_patnos_df.sort_values(by=['INFODT'])
    bl_missing_patnos_df = bl_missing_patnos_df.drop_duplicates(subset=['PATNO'], keep='first')
    bl_baseline_df = pd.concat([bl_baseline_df, bl_missing_patnos_df])
    assert len(bl_baseline_df) == bl_baseline_df.PATNO.nunique()
    bl_baseline_df.to_csv(baseline_filename, index=False)
    
    output_str = cohort_name + '\n'
    output_str += str(df.PATNO.nunique()) + ' patients\n'
    agg_stats = [df.PATNO.nunique()]
    agg_stat_names = ['# patients']
    output_str += str(len(screening_cols)) + ' screening features, ' + str(len(baseline_cols)) + ' baseline features, ' \
        + str(len(total_cols)) + ' assessment totals across time, ' + str(len(symptom_cols)) \
        + ' assessment questions across time, and ' + str(len(shared_cols)) + ' other features across time\n'
    
    output_str += 'Screening features: binary frequency or {10th percentile, mean, 90th percentile}, # patients\n'
    screening_output_str, screening_agg_stats, screening_agg_stat_names = output_baseline_features(sc_baseline_df, screening_cols)
    output_str += screening_output_str
    agg_stats += screening_agg_stats
    agg_stat_names += screening_agg_stat_names
    
    output_str += 'Baseline features: binary frequency or {10th percentile, mean, 90th percentile}, # patients\n'
    baseline_output_str, baseline_agg_stats, baseline_agg_stat_names = output_baseline_features(bl_baseline_df, baseline_cols)
    output_str += baseline_output_str
    agg_stats += baseline_agg_stats
    agg_stat_names += baseline_agg_stat_names
    
    output_str += 'Assessment totals across time: binary frequency or {10th percentile, mean, 90th percentile}, # patients, avg # visits per patient\n'
    total_output_str, total_agg_stats, total_agg_stat_names = output_changing_features(total_df, total_cols)
    output_str += total_output_str
    agg_stats += total_agg_stats
    agg_stat_names += total_agg_stat_names
    
    output_str += 'Assessment questions across time: binary frequency or {10th percentile, mean, 90th percentile}, # patients, avg # visits per patient\n'
    symptom_output_str, symptom_agg_stats, symptom_agg_stat_names = output_changing_features(symptom_df, symptom_cols)
    output_str += symptom_output_str
    agg_stats += symptom_agg_stats
    agg_stat_names += symptom_agg_stat_names
    
    output_str += 'Other features across time: binary frequency or {10th percentile, mean, 90th percentile}, # patients, avg # visits per patient\n'
    other_output_str, other_agg_stats, other_agg_stat_names = output_changing_features(shared_df, shared_cols)
    output_str += other_output_str
    agg_stats += other_agg_stats
    agg_stat_names += other_agg_stat_names
    
    agg_stat_df = pd.DataFrame(agg_stat_names, columns=['Stats'])
    agg_stat_df[cohort_name] = agg_stats
    
    summ_stat_file = output_dir + cohort_name + '_summary.txt'
    with open(summ_stat_file, 'w') as f:
        f.write(output_str)
    
    return agg_stat_df

# separate into 1 csv output per cohort: poss_cohorts = {'HC', 'SWEDD', 'PD', 'GENPD', 'GENUN', 'REGPD', 'REGUN', 'PRODROMA'}
patient_status_path = raw_datadir + 'Patient_Status.csv'
patient_status_df = pd.read_csv(patient_status_path)
enrolled_patient_status_df = patient_status_df.loc[~patient_status_df['ENROLL_DATE'].isnull()][['PATNO','ENROLL_CAT']]
pd_cohort_df = enrolled_patient_status_df.loc[enrolled_patient_status_df['ENROLL_CAT']=='PD']
swedd_cohort_df = enrolled_patient_status_df.loc[enrolled_patient_status_df['ENROLL_CAT']=='SWEDD']
hc_cohort_df = enrolled_patient_status_df.loc[enrolled_patient_status_df['ENROLL_CAT']=='HC']
genpd_cohort_df = enrolled_patient_status_df.loc[enrolled_patient_status_df['ENROLL_CAT']=='GENPD']
genun_cohort_df = enrolled_patient_status_df.loc[enrolled_patient_status_df['ENROLL_CAT']=='GENUN']
regpd_cohort_df = enrolled_patient_status_df.loc[enrolled_patient_status_df['ENROLL_CAT']=='REGPD']
regun_cohort_df = enrolled_patient_status_df.loc[enrolled_patient_status_df['ENROLL_CAT']=='REGUN']
prodroma_cohort_df = enrolled_patient_status_df.loc[enrolled_patient_status_df['ENROLL_CAT']=='PRODROMA']

agg_stat_df = output_df(collected_data.loc[collected_data['PATNO'].isin(pd_cohort_df.PATNO.unique())], 'PD')
swedd_stat_df = output_df(collected_data.loc[collected_data['PATNO'].isin(swedd_cohort_df.PATNO.unique())], 'SWEDD')
agg_stat_df['SWEDD'] = swedd_stat_df['SWEDD']
hc_stat_df = output_df(collected_data.loc[collected_data['PATNO'].isin(hc_cohort_df.PATNO.unique())], 'HC')
agg_stat_df['HC'] = hc_stat_df['HC']
genpd_stat_df = output_df(collected_data.loc[collected_data['PATNO'].isin(genpd_cohort_df.PATNO.unique())], 'GENPD')
agg_stat_df['GENPD'] = genpd_stat_df['GENPD']
genun_stat_df = output_df(collected_data.loc[collected_data['PATNO'].isin(genun_cohort_df.PATNO.unique())], 'GENUN')
agg_stat_df['GENUN'] = genun_stat_df['GENUN']
regpd_stat_df = output_df(collected_data.loc[collected_data['PATNO'].isin(regpd_cohort_df.PATNO.unique())], 'REGPD')
agg_stat_df['REGPD'] = regpd_stat_df['REGPD']
regun_stat_df = output_df(collected_data.loc[collected_data['PATNO'].isin(regun_cohort_df.PATNO.unique())], 'REGUN')
agg_stat_df['REGUN'] = regun_stat_df['REGUN']
prodroma_stat_df = output_df(collected_data.loc[collected_data['PATNO'].isin(prodroma_cohort_df.PATNO.unique())], 'PRODROMA')
agg_stat_df['PRODROMA'] = prodroma_stat_df['PRODROMA']
prodroma_nonmotor_stat_df = output_df(collected_data.loc[np.logical_and(collected_data['PATNO'].isin(prodroma_cohort_df.PATNO.unique()), collected_data['PRODROMAL_DIAG:NONMOTOR_PRODROMA']==1)], 'PRODROMA_NONMOTOR')
agg_stat_df['PRODROMA_NONMOTOR'] = prodroma_nonmotor_stat_df['PRODROMA_NONMOTOR']
prodroma_motor_stat_df = output_df(collected_data.loc[np.logical_and(collected_data['PATNO'].isin(prodroma_cohort_df.PATNO.unique()), collected_data['PRODROMAL_DIAG:MOTOR_PRODROMA']==1)], 'PRODROMA_MOTOR')
agg_stat_df['PRODROMA_MOTOR'] = prodroma_motor_stat_df['PRODROMA_MOTOR']
prodroma_phenoconv_stat_df = output_df(collected_data.loc[np.logical_and(collected_data['PATNO'].isin(prodroma_cohort_df.PATNO.unique()), collected_data['PRODROMAL_DIAG:PHENOCONV']==1)], 'PRODROMA_PHENOCONV')
agg_stat_df['PRODROMA_PHENOCONV'] = prodroma_phenoconv_stat_df['PRODROMA_PHENOCONV']
prodroma_noneuro_stat_df = output_df(collected_data.loc[np.logical_and(collected_data['PATNO'].isin(prodroma_cohort_df.PATNO.unique()), collected_data['PRODROMAL_DIAG:NO_NEURO']==1)], 'PRODROMA_NO_NEURO')
agg_stat_df['PRODROMA_NO_NEURO'] = prodroma_noneuro_stat_df['PRODROMA_NO_NEURO']
agg_stat_df.to_csv(output_dir + 'agg_stat.csv', index=False)