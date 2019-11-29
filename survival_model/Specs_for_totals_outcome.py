import numpy as np, pandas as pd, pickle, sys, os

'''
Take visit feature inputs data directory as parameter.
'''
if len(sys.argv) != 2:
    print('Expecting path to data directory as parameter.')
pipeline_dir = sys.argv[1]
if pipeline_dir[-1] != '/':
    pipeline_dir += '/'
assert os.path.isdir(pipeline_dir)

above_thresh_percentiles = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
below_thresh_percentiles = [70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

pd_df = pd.read_csv(pipeline_dir + 'PD_totals_across_time.csv')
pd_df['NUPDRS3'] = np.nanmax(pd_df[['NUPDRS3_untreated', 'NUPDRS3_on', 'NUPDRS3_off', 'NUPDRS3_unclear']], axis=1)
pd_df['STAI'] = pd_df[['STATE_ANXIETY', 'TRAIT_ANXIETY']].sum(axis=1)

'''
Specify the features in each category and their direction.
'''
motor_dict = {'NUPDRS2': {'NUPDRS2': (np.unique([np.nanpercentile(pd_df['NUPDRS2'].values, thresh) \
                                                 for thresh in above_thresh_percentiles]), True)}, \
              'NUPDRS3': {'NUPDRS3': (np.unique([np.nanpercentile(pd_df['NUPDRS3'].values, thresh) \
                                                 for thresh in above_thresh_percentiles]), True)}}
cog_dict = {'MOCA': {'MOCA': (np.unique([np.nanpercentile(pd_df['MOCA'].values, thresh) \
                                         for thresh in below_thresh_percentiles]), False)}, \
            'BJLO': {'BJLO': (np.unique([np.nanpercentile(pd_df['BJLO'].values, thresh) \
                                         for thresh in below_thresh_percentiles]), False)}, \
            'LNS': {'LNS': (np.unique([np.nanpercentile(pd_df['LNS'].values, thresh) \
                                       for thresh in below_thresh_percentiles]), False)}, \
            'SEMANTIC_FLUENCY': {'SEMANTIC_FLUENCY': (np.unique([np.nanpercentile(pd_df['SEMANTIC_FLUENCY'].values, thresh) \
                                                                 for thresh in below_thresh_percentiles]), False)}, \
            'HVLT_immed_recall': {'HVLT_immed_recall': (np.unique([np.nanpercentile(pd_df['HVLT_immed_recall'].values, thresh) \
                                                                   for thresh in below_thresh_percentiles]), False)}, \
            'HVLT_discrim_recog': {'HVLT_discrim_recog': (np.unique([np.nanpercentile(pd_df['HVLT_discrim_recog'].values, \
                                                                                      thresh) \
                                                                     for thresh in below_thresh_percentiles]), False)}, \
            'HVLT_retent': {'HVLT_retent': (np.unique([np.nanpercentile(pd_df['HVLT_retent'].values, thresh) \
                                                       for thresh in below_thresh_percentiles]), False)}}
psych_dict = {'GDSSHORT': {'GDSSHORT': (np.unique([np.nanpercentile(pd_df['GDSSHORT'].values, thresh) \
                                                   for thresh in below_thresh_percentiles]), False)}, \
              'QUIP': {'QUIP': ([1,2,3,4], True)}, \
              'STAI': {'STAI': (np.unique([np.nanpercentile(pd_df['STAI'].values, thresh) \
                                           for thresh in above_thresh_percentiles]), True)}}
auto_dict = {'SCOPA-AUT': {'SCOPA-AUT': (np.unique([np.nanpercentile(pd_df['SCOPA-AUT'].values, thresh) \
                                                    for thresh in above_thresh_percentiles]), True)}}
sleep_dict = {'EPWORTH': {'EPWORTH': ([1], True)}, \
              'REMSLEEP': {'REMSLEEP': ([1], True)}}
all_dict = {'Motor': motor_dict, 'Cognitive': cog_dict, 'Psychiatric': psych_dict, 'Autonomic': auto_dict, 'Sleep': sleep_dict}

'''
Make a human-readable dictionary for printing tables later.
'''
human_readable_dict = {'NUPDRS2': 'MDS-UPDRS II', 'NUPDRS3': 'MDS-UPDRS III', 'MOCA': 'MoCA', 'BJLO': 'BJLO', \
                       'LNS': 'LNS', 'SEMANTIC_FLUENCY': 'Semantic fluency', 'HVLT_immed_recall': 'HVLT immed recall', \
                       'HVLT_discrim_recog': 'HVLT discrim recog', 'HVLT_retent': 'HVLT retent', 'GDSSHORT': 'GDS depression', \
                       'QUIP': 'QUIP (impulsive)', 'STAI': 'STAI anxiety', 'SCOPA-AUT': 'SCOPA-AUT', 'EPWORTH': 'EPWORTH', \
                       'REMSLEEP': 'RBD sleep'}
# just to check I haven't missed anything
assert set(human_readable_dict.keys()) == set([feat for category in all_dict.keys() for group in all_dict[category].keys() \
                                               for feat in all_dict[category][group]])

min_max_dict = dict()
for feat in human_readable_dict.keys():
    min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))

'''
Store in specs directory.
'''
spec_dir = 'survival_outcome_totals/'
if not os.path.isdir(spec_dir):
    os.makedirs(spec_dir)
with open(spec_dir + 'specs.pkl', 'w') as f:
    pickle.dump(all_dict, f)
with open(spec_dir + 'human_readable_dict.pkl', 'w') as f:
    pickle.dump(human_readable_dict, f)
with open(spec_dir + 'min_max_dict.pkl', 'w') as f:
    pickle.dump(min_max_dict, f)