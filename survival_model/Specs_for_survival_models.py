import numpy as np, pandas as pd, os, pickle

baseline_df = pd.read_csv('../gather_PD_data/selected_baseline_data_using_CMEDTM_modified.csv')
del baseline_df['ENROLL_CAT']
longitudinal_df = pd.read_csv('../gather_PD_data/selected_longitudinal_data_using_CMEDTM_modified.csv')
screening_longitudinal_df = longitudinal_df.loc[longitudinal_df['EVENT_ID_DUR']==0]
baseline_longitudinal_df = longitudinal_df.loc[longitudinal_df['EVENT_ID_DUR']==0.125]
screening_longitudinal_cols = ['NUPDRS1', 'MOCA', 'NUPDRS2_DAILYACT', 'NUPDRS3_GAIT', 'NUPDRS3_RIGID_RIGHT', \
                               'NUPDRS3_FACE', 'NUPDRS3_TREMOR', 'NUPDRS3_RIGID_LEFT']
baseline_longitudinal_cols = ['SCOPA-AUT', 'HVLT_discrim_recog', 'STAI', 'HVLT_immed_recall', 'QUIP', 'EPWORTH', \
                              'GDSSHORT', 'HVLT_retent', 'BJLO', 'LNS', 'SEMANTIC_FLUENCY', 'REMSLEEP']
baseline_df = baseline_df.merge(screening_longitudinal_df[['PATNO']+screening_longitudinal_cols], on=['PATNO'], \
                                validate='one_to_one')
baseline_df = baseline_df.merge(baseline_longitudinal_df[['PATNO']+baseline_longitudinal_cols], on=['PATNO'], \
                                validate='one_to_one')
baseline_df.to_csv('survival_baseline_data.csv', index=False)
print(baseline_df.columns.values)

# spec only most relevant: age, gender, UPSIT, genetic risk score, assessments belonging to each category
selected_baseline_feats = ['AGE', 'MALE', 'UPSIT', 'GENETIC_RISK_SCORE']
selected_feats_dict = dict()
selected_motor_feats = ['NUPDRS2_DAILYACT', 'NUPDRS3_GAIT', 'NUPDRS3_RIGID_RIGHT', 'NUPDRS3_FACE', 'NUPDRS3_TREMOR', \
                        'NUPDRS3_RIGID_LEFT']
selected_feats_dict['Motor'] = selected_baseline_feats + selected_motor_feats
selected_cognitive_feats = ['MOCA', 'HVLT_discrim_recog', 'HVLT_immed_recall', 'HVLT_retent', 'BJLO', 'LNS', 'SEMANTIC_FLUENCY']
selected_feats_dict['Cognitive'] = selected_baseline_feats + selected_cognitive_feats
selected_autonomic_feats = ['SCOPA-AUT']
selected_feats_dict['Autonomic'] = selected_baseline_feats + selected_autonomic_feats
selected_psychiatric_feats = ['STAI', 'QUIP', 'GDSSHORT']
selected_feats_dict['Psychiatric'] = selected_baseline_feats + selected_psychiatric_feats
selected_sleep_feats = ['EPWORTH', 'REMSLEEP']
selected_feats_dict['Sleep'] = selected_baseline_feats + selected_sleep_feats
selected_feats_dict['hybrid_requiremotor'] = selected_baseline_feats + selected_motor_feats + selected_cognitive_feats \
    + selected_autonomic_feats + selected_psychiatric_feats + selected_sleep_feats

spec_dir = 'hand_selected_feats_models/'
if not os.path.isdir(spec_dir):
    os.makedirs(spec_dir)
with open(spec_dir + 'selected_baseline_feats.pkl', 'w') as f:
    pickle.dump(selected_feats_dict, f)