import numpy as np, pandas as pd, pickle, sys, os

def compute_subtotals_df(pd_questions_df):
    subtotal_maps = {'NUPDRS3_TREMOR': ['NP3RTALL', 'NP3RTALU', 'NP3KTRML', 'NP3PTRML', 'NP3KTRMR', 'NP3PTRMR', 'NP3RTARU', \
                                        'NP3RTALJ', 'NP3RTARL', 'NP2TRMR', 'NP3RTCON'], \
                     'NUPDRS3_RIGID_LEFT': ['NP3RIGLU', 'NP3RIGLL', 'NP3PRSPL', 'NP3FTAPL', 'NP3HMOVL', 'NP3LGAGL', \
                                            'NP3TTAPL'], \
                     'NUPDRS3_RIGID_RIGHT': ['NP3RIGRL', 'NP3RIGRU', 'NP3PRSPR', 'NP3FTAPR', 'NP3HMOVR', 'NP3LGAGR', \
                                             'NP3TTAPR'], \
                     'NUPDRS3_FACE': ['NP3SPCH', 'NP3RIGN', 'NP3BRADY', 'NP3FACXP'], \
                     'NUPDRS3_GAIT': ['NP3FRZGT', 'NP3PSTBL', 'NP3RISNG', 'NP3GAIT', 'NP3POSTR'], \
                     'NUPDRS2_DAILYACT': ['NP2HWRT', 'NP2FREZ', 'NP2HYGN', 'NP2EAT', 'NP2HOBB', 'NP2WALK', 'NP2DRES', \
                                          'NP2RISE', 'NP2TURN', 'NP2SWAL', 'NP2SALV', 'NP2SPCH']
                    }
    for subtotal in subtotal_maps.keys():
        untreated_cols = []
        on_cols = []
        off_cols = []
        maob_cols = []
        for feat in subtotal_maps[subtotal]:
            if feat.startswith('NP2'):
                untreated_cols.append(feat)
                on_cols.append(feat)
                off_cols.append(feat)
                maob_cols.append(feat)
            else:
                untreated_cols.append(feat + '_untreated')
                on_cols.append(feat + '_on')
                off_cols.append(feat + '_off')
                maob_cols.append(feat + '_maob')
        pd_questions_df[subtotal + '_untreated'] \
            = np.where(pd.isnull(pd_questions_df[untreated_cols[0]]), float('NaN'), \
                       pd_questions_df[untreated_cols].sum(axis=1))
        pd_questions_df[subtotal + '_on'] \
            = np.where(pd.isnull(pd_questions_df[on_cols[0]]), float('NaN'), \
                       pd_questions_df[on_cols].sum(axis=1))
        pd_questions_df[subtotal + '_off'] \
            = np.where(pd.isnull(pd_questions_df[off_cols[0]]), float('NaN'), \
                       pd_questions_df[off_cols].sum(axis=1))
        pd_questions_df[subtotal + '_maob'] \
            = np.where(pd.isnull(pd_questions_df[maob_cols[0]]), float('NaN'), \
                       pd_questions_df[maob_cols].sum(axis=1))
        pd_questions_df[subtotal] = np.nanmax(pd_questions_df[[subtotal + '_untreated', subtotal + '_on', subtotal + '_off', \
                                                               subtotal + '_maob']], axis=1)
    return pd_questions_df

def main():
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
    pd_df['NUPDRS3'] = np.nanmax(pd_df[['NUPDRS3_untreated', 'NUPDRS3_on', 'NUPDRS3_off', 'NUPDRS3_maob']], axis=1)
    pd_df['STAI'] = pd_df[['STATE_ANXIETY', 'TRAIT_ANXIETY']].sum(axis=1)

    subtotal_maps = {'NUPDRS3_TREMOR': ['NP3RTALL', 'NP3RTALU', 'NP3KTRML', 'NP3PTRML', 'NP3KTRMR', 'NP3PTRMR', 'NP3RTARU', \
                                        'NP3RTALJ', 'NP3RTARL', 'NP2TRMR', 'NP3RTCON'], \
                     'NUPDRS3_RIGID_LEFT': ['NP3RIGLU', 'NP3RIGLL', 'NP3PRSPL', 'NP3FTAPL', 'NP3HMOVL', 'NP3LGAGL', \
                                            'NP3TTAPL'], \
                     'NUPDRS3_RIGID_RIGHT': ['NP3RIGRL', 'NP3RIGRU', 'NP3PRSPR', 'NP3FTAPR', 'NP3HMOVR', 'NP3LGAGR', \
                                             'NP3TTAPR'], \
                     'NUPDRS3_FACE': ['NP3SPCH', 'NP3RIGN', 'NP3BRADY', 'NP3FACXP'], \
                     'NUPDRS3_GAIT': ['NP3FRZGT', 'NP3PSTBL', 'NP3RISNG', 'NP3GAIT', 'NP3POSTR'], \
                     'NUPDRS2_DAILYACT': ['NP2HWRT', 'NP2FREZ', 'NP2HYGN', 'NP2EAT', 'NP2HOBB', 'NP2WALK', 'NP2DRES', \
                                          'NP2RISE', 'NP2TURN', 'NP2SWAL', 'NP2SALV', 'NP2SPCH']
                    }
    pd_questions_df = pd.read_csv(pipeline_dir + 'PD_questions_across_time.csv')

    pd_questions_df = compute_subtotals_df(pd_questions_df)
    pd_df = pd_df.merge(pd_questions_df[['PATNO','EVENT_ID_DUR']+subtotal_maps.keys()].dropna(), on=['PATNO','EVENT_ID_DUR'], \
                        how='outer', validate='one_to_one')

    '''
    Specify the features in each category and their direction.
    '''
    motor_dict = dict()
    for subtotal in subtotal_maps.keys():
        motor_dict[subtotal] = {subtotal: (np.unique([np.nanpercentile(pd_df[subtotal].values, thresh) \
                                                      for thresh in above_thresh_percentiles]), True)}
    cog_dict = {'MOCA': {'MOCA': (np.unique([np.nanpercentile(pd_df['MOCA'].values, thresh) \
                                             for thresh in below_thresh_percentiles]), False)}, \
                'BJLO': {'BJLO': (np.unique([np.nanpercentile(pd_df['BJLO'].values, thresh) \
                                             for thresh in below_thresh_percentiles]), False)}, \
                'LNS': {'LNS': (np.unique([np.nanpercentile(pd_df['LNS'].values, thresh) \
                                           for thresh in below_thresh_percentiles]), False)}, \
                'SEMANTIC_FLUENCY': {'SEMANTIC_FLUENCY': (np.unique([np.nanpercentile(pd_df['SEMANTIC_FLUENCY'].values, thresh) \
                                                                     for thresh in below_thresh_percentiles]), False)}, \
                'HVLT_immed_recall': {'HVLT_immed_recall': \
                                      (np.unique([np.nanpercentile(pd_df['HVLT_immed_recall'].values, thresh) \
                                                  for thresh in below_thresh_percentiles]), False)}, \
                'HVLT_discrim_recog': {'HVLT_discrim_recog': (np.unique([np.nanpercentile(pd_df['HVLT_discrim_recog'].values, \
                                                                                          thresh) \
                                                                         for thresh in below_thresh_percentiles]), False)}, \
                'HVLT_retent': {'HVLT_retent': (np.unique([np.nanpercentile(pd_df['HVLT_retent'].values, thresh) \
                                                           for thresh in below_thresh_percentiles]), False)}}
    #psych_dict = {'GDSSHORT': {'GDSSHORT': (np.unique([np.nanpercentile(pd_df['GDSSHORT'].values, thresh) \
    #                                                   for thresh in below_thresh_percentiles]), False)}, \
    psych_dict = {'GDSSHORT': {'GDSSHORT': ([1], True)}, \
                  'QUIP': {'QUIP': ([1,2,3,4], True)}, \
                  'STAI': {'STAI': (np.unique([np.nanpercentile(pd_df['STAI'].values, thresh) \
                                               for thresh in above_thresh_percentiles]), True)}}
    auto_dict = {'SCOPA-AUT': {'SCOPA-AUT': (np.unique([np.nanpercentile(pd_df['SCOPA-AUT'].values, thresh) \
                                                        for thresh in above_thresh_percentiles]), True)}}
    sleep_dict = {'EPWORTH': {'EPWORTH': ([1], True)}, \
                  'REMSLEEP': {'REMSLEEP': ([1], True)}}
    all_dict = {'Motor': motor_dict, 'Cognitive': cog_dict, 'Psychiatric': psych_dict, 'Autonomic': auto_dict, \
                'Sleep': sleep_dict}

    '''
    Make a human-readable dictionary for printing tables later.
    '''
    human_readable_dict = {'NUPDRS3_TREMOR': 'MDS-UPDRS III tremor', 'NUPDRS3_RIGID_LEFT': 'MDS-UPDRS III left rigidity', \
                           'NUPDRS3_RIGID_RIGHT': 'MDS-UPDRS III right rigidity', 'NUPDRS3_FACE': 'MDS-UPDRS III face', \
                           'NUPDRS3_GAIT': 'MDS-UPDRS III gait', 'NUPDRS2_DAILYACT': 'MDS-UPDRS II daily activities', \
                           'MOCA': 'MoCA', 'BJLO': 'BJLO', 'LNS': 'LNS', 'SEMANTIC_FLUENCY': 'Semantic fluency', \
                           'HVLT_immed_recall': 'HVLT immed recall', 'HVLT_discrim_recog': 'HVLT discrim recog', \
                           'HVLT_retent': 'HVLT retent', 'GDSSHORT': 'GDS depression', \
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
    spec_dir = 'survival_outcome_subtotals_gdsfixed_using_CMEDTM/'
    if not os.path.isdir(spec_dir):
        os.makedirs(spec_dir)
    with open(spec_dir + 'specs.pkl', 'w') as f:
        pickle.dump(all_dict, f)
    with open(spec_dir + 'human_readable_dict.pkl', 'w') as f:
        pickle.dump(human_readable_dict, f)
    with open(spec_dir + 'min_max_dict.pkl', 'w') as f:
        pickle.dump(min_max_dict, f)
        
if __name__ == '__main__':
    main()
