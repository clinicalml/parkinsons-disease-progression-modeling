import numpy as np, pandas as pd, pickle, sys, os

'''
Take visit feature inputs data directory as parameter.
'''
if len(sys.argv) != 2:
    print('Expecting path to data directory as parameter.')
pipeline_dir = sys.argv[1]
if pipeline_dir[-1] != '/':
    pipeline_dir += '/'

human_readable_dict = {'NP2DRES': 'MDS-UPDRS 2.5 dressing', 'NP2EAT': 'MDS-UPDRS 2.4 eating tasks', \
                       'NP2HYGN': 'MDS-UPDRS 2.6 hygiene', 'NP2WALK': 'MDS-UPDRS 2.12 walking and balance', \
                       'NP2HOBB': 'MDS-UPDRS 2.8 hobbies and other activities', \
                       'NP2RISE': 'MDS-UPDRS  2.11 getting out of bed/car/deep chair', \
                       'NP2TURN': 'MDS-UPDRS 2.9 turning in bed', 'NP2SALV': 'MDS-UPDRS 2.2 saliva and drooling', \
                       'NP2FREZ': 'MDS-UPDRS 2.13 freezing', 'NP2HWRT': 'MDS-UPDRS 2.7 handwriting', \
                       'NP2SPCH': 'MDS-UPDRS 2.1 speech', 'NP2TRMR': 'MDS-UPDRS 2.10 tremor', \
                       'NP2SWAL': 'MDS-UPDRS 2.3 chewing and swallowing',\
                       
                       'NP3RIGLU': 'MDS-UPDRS 3.3 rigidity upper left', 'NP3FACXP': 'MDS-UPDRS 3.2 facial expression', \
                       'NP3RIGRL': 'MDS-UPDRS 3.3 rigidity right lower', \
                       'NP3KTRML': 'MDS-UPDRS 3.16 kinetic tremor of hands left', \
                       'NP3FTAPR': 'MDS-UPDRS 3.4 finger tapping right', \
                       'NP3RTCON': 'MDS-UPDRS 3.18 constancy of rest tremor', \
                       'NP3PRSPL': 'MDS-UPDRS 3.6 pronation-supination movements of hands left', \
                       'NP3SPCH': 'MDS-UPDRS 3.1 speech', 'NP3RTALJ': 'MDS-UPDRS 3.17 rest tremor amplitude lip/jaw', \
                       'NP3LGAGR': 'MDS-UPDRS 3.8 leg agility right', 'NP3HMOVR': 'MDS-UPDRS 3.5 hand movements right', \
                       'NP3RISNG': 'MDS-UPDRS 3.9 arising from chair', \
                       'NP3RTALU': 'MDS-UPDRS 3.17 rest tremor amplitude left upper', \
                       'NP3FTAPL': 'MDS-UPDRS 3.4 finger tapping left', \
                       'NP3RTALL': 'MDS-UPDRS 3.17 rest tremor amplitude left lower', \
                       'NP3RIGRU': 'MDS-UPDRS 3.3 rigidity right upper', 'NP3TTAPL': 'MDS-UPDRS 3.7 toe tapping left', \
                       'NP3PSTBL': 'MDS-UPDRS 3.12 postural stability', \
                       'NP3RTARL': 'MDS-UPDRS 3.17 rest tremor amplitude right lower', \
                       'NP3BRADY': 'MDS-UPDRS 3.14 global spontaneity of movement/body bradykinesia', \
                       'NP3RTARU': 'MDS-UPDRS 3.17 rest tremor amplitude right upper', \
                       'NP3PRSPR': 'MDS-UPDRS 3.6 pronation-supination movements of hands right', \
                       'NP3RIGN': 'MDS-UPDRS 3.3 rigidity neck', 'NP3RIGLL': 'MDS-UPDRS 3.3 rigidity lower left', \
                       'NP3PTRML': 'MDS-UPDRS 3.15 postural tremor of hands left', \
                       'NP3TTAPR': 'MDS-UPDRS 3.7 toe tapping right', 'NP3LGAGL': 'MDS-UPDRS 3.8 leg agility left', \
                       'NP3HMOVL': 'MDS-UPDRS 3.5 hand movements left', 'NP3GAIT': 'MDS-UPDRS 3.10 gait', \
                       'NP3PTRMR': 'MDS-UPDRS 3.15 postural tremor of hands right', \
                       'NP3KTRMR': 'MDS-UPDRS 3.16 kinetic tremor of hands right', \
                       'NP3POSTR': 'MDS-UPDRS 3.13 posture', 'NP3FRZGT': 'MDS-UPDRS 3.11 freezing of gait', \
                       
                       'MCARHINO': 'MoCA name rhino from drawing', 'MCADATE': 'MoCA today\'s date', \
                       'MCADAY': 'MoCA today\'s day', 'MCAVF': 'MoCA come up with words starting with F', \
                       'MCAFDS': 'MoCA repeat numbers in forward order', 'MCAPLACE': 'MoCA name of place', \
                       'MCAREC1': 'MoCA recall word 1', 'MCAREC2': 'MoCA recall word 2', 'MCAREC3': 'MoCA recall word 3', \
                       'MCAREC4': 'MoCA recall word 4', 'MCAREC5': 'MoCA recall word 5', \
                       'MCAVIGIL': 'MoCA tap foot on each A', 'MCACLCKN': 'MoCA draw clock numbers', \
                       'MCACLCKH': 'MoCA draw clock hands', 'MCACLCKC': 'MoCA draw clock contour', \
                       'MCAALTTM': 'MoCA top left question connect 1 $\\rightarrow$ A $\\rightarrow$ 2 $\\rightarrow$ ...', \
                       'MCACITY': 'MoCA name of city', 'MCASER7': 'MoCA serial 7 subtraction starting at 100', \
                       'MCALION': 'MoCA name lion from drawing', 'MCABDS': 'MoCA repeat numbers in backward order', \
                       'MCAABSTR': 'MoCA abstraction', 'MCACAMEL': 'MoCA name camel from drawing', \
                       'MCACUBE': 'MoCA copy cube', 'MCAYR': 'MoCA today\'s year', 'MCASNTNC': 'MoCA repeat sentences', \
                       'MCAMONTH': 'MoCA today\'s month', \
                       
                       'VLTANIM': 'Semantic fluency (animals)', 'VLTVEG': 'Semantic fluency (vegetables)', \
                       'VLTFRUIT': 'Semantic fluency (fruits)', \
                       
                       'HVLT_immed_recall': 'HVLT immed recall', 'HVLT_discrim_recog': 'HVLT discrim recog', \
                       'HVLT_retent': 'HVLT retent', \
                       
                       'COGSTATE': 'Cognitive categorization (0=Normal, 1=MCI, 2=dementia)', \
                       'COGDECLN': 'Cognitive decline', \
                       'FNCDTCOG': 'Cognitive function deterioration', \
                       'DVT_SDM': 'Symbol digits modality', \
                       
                       'GDSAFRAD': 'GDS afraid bad happen', 'GDSHOPLS': 'GDS hopeless', 'GDSHLPLS': 'GDS helpless', \
                       'GDSHAPPY': 'GDS happy', 'GDSBORED': 'GDS bored', 'GDSMEMRY': 'GDS memory problems', \
                       'GDSSATIS': 'GDS satisfied with life', 'GDSEMPTY': 'GDS empty life', \
                       'GDSBETER': 'GDS most better off than self', 'GDSGSPIR': 'GDS good spirits', \
                       'GDSENRGY': 'GDS full of energy', 'GDSDROPD': 'GDS dropped interests', \
                       'GDSWRTLS': 'GDS feel worthless', 'GDSHOME': 'GDS prefer stay home', \
                       'GDSALIVE': 'GDS wonderful alive', \
                       
                       'TMGAMBLE': 'QUIP too much gambling', 'CNTRLGMB': 'QUIP can\'t control gambling', \
                       'TMSEX': 'QUIP too much sex behaviors', \
                       'CNTRLSEX': 'QUIP think too much about sex behaviors', 'TMBUY': 'QUIP too much buying', \
                       'CNTRLBUY': 'QUIP activities to continue buying', \
                       'TMEAT': 'QUIP too much eating behaviors', \
                       'CNTRLEAT': 'QUIP distressed about eating behaviors', \
                       'TMTORACT': 'QUIP too much time on specific activities', \
                       'TMTMTACT': 'QUIP too much time repeating simple motor activities', \
                       'TMTRWD': 'QUIP too much time walk/drive w/o goal', \
                       'TMDISMED': 'QUIP too much PD med use', \
                       'CNTRLDSM': 'QUIP can\'t control PD med use', \
                       
                       'STAIAD1': 'STAI 1 calm', 'STAIAD2': 'STAI 2 secure', 'STAIAD3': 'STAI 3 tense', \
                       'STAIAD4': 'STAI 4 strained', 'STAIAD5': 'STAI 5 at ease', 'STAIAD6': 'STAI 6 upset', \
                       'STAIAD7': 'STAI 7 worry possible misfortunes', 'STAIAD8': 'STAI 8 satisfied', \
                       'STAIAD9': 'STAI 9 frightened', 'STAIAD10': 'STAI 10 comfortable', \
                       'STAIAD11': 'STAI 11 self-confident', 'STAIAD12': 'STAI 12 nervous', 'STAIAD13': 'STAI 13 jittery', \
                       'STAIAD14': 'STAI 14 indecisive', 'STAIAD15': 'STAI 15 relaxed', 'STAIAD16': 'STAI 16 content', \
                       'STAIAD17': 'STAI 17 worried', 'STAIAD18': 'STAI 18 confused', 'STAIAD19': 'STAI 19 steady', \
                       'STAIAD20': 'STAI 20 pleasant', 'STAIAD21': 'STAI 21 pleasant', \
                       'STAIAD22': 'STAI 22 nervous + restless', 'STAIAD23': 'STAI 23 satisfied with myself', \
                       'STAIAD24': 'STAI 24 wish could be happy', 'STAIAD25': 'STAI 25 feel like failure', \
                       'STAIAD26': 'STAI 26 rested', 'STAIAD27': 'STAI 27 calm/cool/collected', \
                       'STAIAD28': 'STAI 28 can\'t overcome difficulties', \
                       'STAIAD29':  'STAI 29 worry too much over trivial', 'STAIAD30': 'STAI 30 happy', \
                       'STAIAD31': 'STAI 31 disturbing thoughts', 'STAIAD32': 'STAI 32 lack self-confidence', \
                       'STAIAD33': 'STAI 33 secure', 'STAIAD34': 'STAI 34 make decisions easily', \
                       'STAIAD35': 'STAI 35 inadequate', 'STAIAD36': 'STAI 36 content', \
                       'STAIAD37': 'STAI 37 unimportant thought bothers me', \
                       'STAIAD38': 'STAI 38 feel disappointments strongly', 'STAIAD39': 'STAI steady', \
                       'STAIAD40': 'STAI 40 tension/turmoil', \
                       
                       'DRMVIVID': 'vivid dreams (REM 1)', 'DRMAGRAC': 'aggressive/action dreams (REM 2)', \
                       'DRMNOCTB': 'dream contents match nocturnal behavior (REM 3)', \
                       'SLPLMBMV': 'arms/legs move in sleep (REM 4)', 'SLPINJUR': 'injured self/partner in sleep (REM 5)', \
                       'DRMVERBL': 'speak in dreams (REM 6.1)', 'DRMFIGHT': 'sudden move in dreams (REM 6.2)', \
                       'DRMUMV': 'gestures in dreams (REM 6.3)', 'DRMOBJFL': 'things near bed fall in dreams (REM 6.4)', \
                       'MVAWAKEN': 'movements wake patient (REM 7)', 'DRMREMEM': 'remember dreams (REM 8)', \
                       'SLPDSTRB': 'disturbed sleep (REM 9)', 'STROKE': 'had stroke (REM 10.a)', \
                       'HETRA': 'had head trauma (REM 10.b)', 'PARKISM': 'had parkinsonism (REM 10.c)', \
                       'RLS': 'had restless leg syndrome (REM 10.d)', 'NARCLPSY': 'had narcolepsy (REM 10.e)', \
                       'DEPRS': 'had depression (REM 10.f)', 'EPILEPSY': 'had epilepsy (REM 10.g)', \
                       'BRNINFM': 'had brain inflammation (REM 10.h)', \
                       'CNSOTH': 'had other nervous system disease (REM 10.i)', \
                       
                       'ESS1': 'sleep when sit + read', 'ESS2': 'sleep when watch TV', \
                       'ESS3': 'sleep when sit in public', 'ESS4': 'sleep on car for 1hr', \
                       'ESS5': 'sleep when lie down afternoon', 'ESS6': 'sleep when talk', 'ESS7': 'sleep after lunch', \
                       'ESS8': 'sleep on car stopped a few min', \
                       
                       'SCAU1': 'SCOPA swallow difficult/choke', 'SCAU2': 'SCOPA drool', 'SCAU3': 'SCOPA food stuck', \
                       'SCAU4': 'SCOPA full quickly', 'SCAU5': 'SCOPA constipation', 'SCAU6': 'SCOPA strain pass stool', \
                       'SCAU7': 'SCOPA involuntary stool', 'SCAU8': 'SCOPA difficult retain urine', \
                       'SCAU9': 'SCOPA involuntary urine', 'SCAU10': 'SCOPA can\'t completely empty bladder', \
                       'SCAU11': 'SCOPA urine weak', 'SCAU12': 'SCOPA urine again w/in 2hrs', \
                       'SCAU13': 'SCOPA urine at night', 'SCAU14': 'SCOPA lightheaded upon standing', \
                       'SCAU15': 'SCOPA lightheaded after stand some time', 'SCAU16': 'SCOPA fainted past 6 months', \
                       'SCAU17': 'SCOPA sweat excess daytime', 'SCAU18': 'SCOPA sweat excess nighttime', \
                       'SCAU19': 'SCOPA light-sensitive', 'SCAU20': 'SCOPA intolerant of cold', \
                       'SCAU21': 'SCOPA intolerant of heat', 'SCAU22': 'SCOPA male impotent', \
                       'SCAU23': 'SCOPA male unable ejaculate', 'SCAU23A': 'SCOPA male erection disorder med', \
                       'SCAU24': 'SCOPA female dry vagina', 'SCAU25': 'SCOPA female orgasm difficult', \
                       'SCAU26A': 'SCOPA constipation med', 'SCAU26B': 'SCOPA urine med', \
                       'SCAU26C': 'SCOPA blood pressure med', 'SCAU_catheter': 'SCOPA use catheter', \
                       
                       'NP1COG': 'MDS-UPDRS 1.1 cognitive', 'NP1HALL': 'MDS-UPDRS 1.2 hallucinations', \
                       'NP1DPRS': 'MDS-UPDRS 1.3 depressed', 'NP1ANXS': 'MDS-UPDRS 1.4 anxious', \
                       'NP1APAT': 'MDS-UPDRS 1.5 apathy', 'NP1DDS': 'MDS-UPDRS 1.6 dopamine dysregulation syndrome', \
                       'NP1SLPN': 'MDS-UPDRS 1.7 nighttime sleep problems', 'NP1SLPD': 'MDS-UPDRS 1.8 daytime sleepiness', \
                       'NP1PAIN': 'MDS-UPDRS 1.9 pain', 'NP1URIN': 'MDS-UPDRS 1.10 urine', \
                       'NP1CNST': 'MDS-UPDRS 1.11 constipation', 'NP1LTHD': 'MDS-UPDRS 1.12 lightheaded', \
                       'NP1FATG': 'MDS-UPDRS 1.13 fatigue', \
                       
                       'HRSUP': 'Heart rate supine', 'HRSTND': 'Heart rate standing', \
                       'DIASTND': 'Blood pressure diastolic', 'SYSSTND': 'Blood pressure systolic', \
                       
                       'DFSTROKE': 'DF stroke risk factors', 'DFRSKFCT': 'DF atypical risk factors', \
                       'DFPRESNT': 'DF atypical symptoms', 'DFRPROG': 'DF rapid progression', 'DFSTATIC': 'DF little change', \
                       'DFHEMPRK': 'DF hemiparkinsonism >6yrs', 'DFAGESX': 'DF onset before 30', \
                       'DFOTHCRS': 'DF other atypical', 'DFRTREMP': 'DF rest tremor present', \
                       'DFRTREMA': 'DF rest tremor absent', 'DFPATREM': 'DF prominent action tremor', \
                       'DFOTHTRM': 'DF other tremor', 'DFRIGIDP': 'DF rigidity present', 'DFRIGIDA': 'DF rigidity absent', \
                       'DFAXRIG': 'DF axial rigidity', 'DFUNIRIG': 'DF unilateral rigidity', 'DFTONE': 'DF increased tone', \
                       'DFOTHRIG': 'DF other rigidity', 'DFBRADYP': 'DF bradykinesia present', \
                       'DFBRADYA': 'DF bradykinesia absent', 'DFAKINES': 'DF akinesia', \
                       'DFBRPLUS': 'DF other rapid move difficulties', 'DFOTHABR': 'DF other a/bradykinesia', \
                       'DFPGDIST': 'DF postural gait disturbances', 'DFGAIT': 'DF wide-based gait', \
                       'DFFREEZ': 'DF freezing', 'DFFALLS': 'DF likely to fall', \
                       'DFOTHPG': 'DF other posture/gait disturbances', 'DFPSYCH': 'DF psychiatric', \
                       'DFCOGNIT': 'DF cognitive', 'DFDYSTON': 'DF dystonia', 'DFCHOREA': 'DF chorea', \
                       'DFMYOCLO': 'DF myoclonus', 'DFOTHHYP': 'DF other hyperkinesias', 'DFHEMTRO': 'DF body hemiatrophy', \
                       'DFPSHYPO': 'DF hypotension', 'DFSEXDYS': 'DF sexual dysfunction', \
                       'DFURDYS': 'DF urinary dysfunction', 'DFBWLDYS': 'DF bowel dysfunction', \
                       'DFOCULO': 'DF oculomotor disturbances', 'DFEYELID': 'DF eyelid disturbances', \
                       'DFNEURAB': 'DF atypical neuro abnormalities', 'DFDOPRSP': 'DF little levodopa response', \
                       'DFRAPSPE': 'DF rapid speech', 'DFBULBAR': 'DF dysphagia', 'DFCTSCAN': 'DF CT suggests other cause', \
                       'DFMRI': 'DF MRI suggests other cause', 'DFATYP': 'DF atypical disease', \
                       
                       'ABETA_log': 'CSF amyloid beta log', 'TTAU_log': 'CSF total Tau log', \
                       'PTAU_log': 'CSF phosphorylated Tau log', 'ASYNU_log': 'CSF alpha-synuclein log', \
                       'PTAU_ABETA_ratio': 'CSF pTau to Abeta ratio', 'TTAU_ABETA_ratio': 'CSF tTau to Abeta ratio', \
                       'PTAU_TTAU_ratio': 'CSF pTau to tTau ratio', \
                       
                       'contralateral_caudate': 'DaTscan contralateral caudate', \
                       'ipsilateral_caudate': ' DaTscan ipsilateral_caudate', \
                       'contralateral_putamen': 'DaTscan contralateral putamen', \
                       'ipsilateral_putamen': 'DaTscan ipsilateral putamen', \
                       'count_density_ratio_contralateral': 'DaTscan count density ratio contralateral', \
                       'count_density_ratio_ipsilateral': 'DaTscan count density ratio ipsilateral', \
                       'asymmetry_index_putamen': 'DaTscan asymmetry index putamen', \
                       'asymmetry_index_caudate': 'DaTscan asymmetry index caudate', \
                       
                       'CN1RSP': 'Cranial nerve 1', 'CN2RSP': 'Cranial nerve 2', 'CN346RSP': 'Cranial nerves 3,4,6', \
                       'CN5RSP': 'Cranial nerve 5', 'CN7RSP': 'Cranial nerve 7', 'CN8RSP': 'Cranial nerve 8', \
                       'CN910RSP': 'Cranial nerves 9,10','CN11RSP': 'Cranial nerve 11','CN12RSP': 'Cranial nerve 12', \
                       'MSRARSP': 'Motor strength right arm', 'MSLARSP': 'Motor strength left arm', \
                       'MSRLRSP': 'Motor strength right leg', 'MSLLRSP': 'Motor strength left leg', \
                       'COFNRRSP': 'Coordination right hand', 'COFNLRSP': 'Coordination left hand', \
                       'COHSRRSP': 'Coordination right leg', 'COHSLRSP': 'Coordination left leg', \
                       'SENRARSP': 'Sensation right arm', 'SENLARSP': 'Sensation left arm', \
                       'SENRLRSP': 'Sensation right leg', 'SENLLRSP': 'Sensation left leg', \
                       'RFLRARSP_hyper': 'Hyperactive muscle stretch reflex right arm', \
                       'RFLLARSP_hyper': 'Hyperactive muscle stretch reflex left arm', \
                       'RFLRLRSP_hyper': 'Hyperactive muscle stretch reflex right leg', \
                       'RFLLLRSP_hyper': 'Hyperactive muscle stretch reflex left leg'
                       
                       }

for i in range(1,31):
    human_readable_dict['BJLOT' + str(i)] = 'BJLO ' + str(i)
for i in range(1,8):
    for letter in ['A', 'B', 'C']:
        human_readable_dict['LNS' + str(i) + letter] = 'LNS ' + str(i) + letter

'''
Read in PD dataframe. Take maximum of MDS-UPDRS part III questions at same visit, regardless of treatment status
'''
def handle_mdsupdrs3(df):
    for col in df.columns:
        if col.endswith('_untreated'):
            col_root = col[:int(-1*len('_untreated'))]
            col_4variants = [col_root + '_untreated', col_root + '_on', col_root + '_off', col_root + '_unclear']
            assert set(col_4variants).issubset(set(df.columns.values.tolist()))
            df[col_root] = np.nanmax(df[col_4variants], axis=1)
    return df
pd_totals_df = pd.read_csv(pipeline_dir + 'PD_totals_across_time.csv')
pd_totals_df = pd_totals_df.dropna(subset=['PATNO','EVENT_ID_DUR'])
pd_totals_df = handle_mdsupdrs3(pd_totals_df)
pd_totals_df['STAI'] = pd_totals_df[['STATE_ANXIETY','TRAIT_ANXIETY']].sum(axis=1)
pd_questions_df = pd.read_csv(pipeline_dir + 'PD_questions_across_time.csv')
pd_questions_df = pd_questions_df.dropna(subset=['PATNO','EVENT_ID_DUR'])
pd_questions_df = handle_mdsupdrs3(pd_questions_df)
pd_other_df = pd.read_csv(pipeline_dir + 'PD_other_across_time.csv')
pd_other_df = pd_other_df.dropna(subset=['PATNO','EVENT_ID_DUR'])
pd_other_df = handle_mdsupdrs3(pd_other_df)
totals_cols_used = set(pd_totals_df.columns.values.tolist()).intersection(human_readable_dict.keys())
questions_cols_used = set(pd_questions_df.columns.values.tolist()).intersection(human_readable_dict.keys())
other_cols_used = set(pd_other_df.columns.values.tolist()).intersection(human_readable_dict.keys())
assert len(totals_cols_used.intersection(questions_cols_used)) == 0
assert len(totals_cols_used.intersection(other_cols_used)) == 0
assert len(questions_cols_used.intersection(other_cols_used)) == 0
assert len(human_readable_dict) >= len(totals_cols_used) + len(questions_cols_used) + len(other_cols_used)
pd_totals_df = pd_totals_df[['PATNO', 'EVENT_ID_DUR'] + list(totals_cols_used)]
pd_questions_df = pd_questions_df[['PATNO', 'EVENT_ID_DUR'] + list(questions_cols_used)]
pd_other_df = pd_other_df[['PATNO', 'EVENT_ID_DUR'] + list(other_cols_used)]
pd_df = pd_totals_df.merge(pd_questions_df, on=['PATNO', 'EVENT_ID_DUR'], how='outer', validate='one_to_one')
pd_df = pd_df.merge(pd_other_df, on=['PATNO', 'EVENT_ID_DUR'], how='outer', validate='one_to_one')

'''
Gather the feature directions, thresholds to try, and groupings.
'''
above_thresh_percentiles = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
below_thresh_percentiles = [70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
spec_dict = dict()
min_max_dict = dict()

def get_feat_threshs_to_try(feat_vals, above_or_below):
    assert above_or_below in {'above', 'below'}
    if above_or_below == 'below':
        feat_threshs_to_try = np.unique([np.nanpercentile(pd_df[feat].values, thresh) for thresh in below_thresh_percentiles])
        if feat_threshs_to_try[0] == np.nanmax(pd_df[feat].values):
            feat_threshs_to_try = feat_threshs_to_try[1:]
    else:
        feat_threshs_to_try = np.unique([np.nanpercentile(pd_df[feat].values, thresh) for thresh in above_thresh_percentiles])
        if feat_threshs_to_try[0] == np.nanmin(pd_df[feat].values):
            feat_threshs_to_try = feat_threshs_to_try[1:]
    return feat_threshs_to_try

motor_group_lookup = {'NP2WALK': 'NP2WALK', 'NP2SPCH': 'NP2SPCH', 'NP2FREZ': 'NP2FREZ', 'NP2TRMR': 'NP2TRMR', \
                      'NP2DRES': 'NP2DRES', 'NP2EAT': 'NP2EAT', 'NP2HYGN': 'NP2HYGN', 'NP2HOBB': 'NP2HOBB', \
                      'NP2RISE': 'NP2RISE', 'NP2TURN': 'NP2TURN', 'NP2SALV': 'NP2SALV', 'NP2HWRT': 'NP2HWRT', \
                      'NP2SWAL': 'NP2SWAL', \
                      
                      'NP3GAIT': 'Gait', 'NP3FRZGT': 'Gait', 'NP3SPCH': 'NP3SPCH', \
                      'NP3PTRMR': 'Postural tremor', 'NP3PTRML': 'Postural tremor', \
                      'NP3KTRMR': 'Kinetic tremor', 'NP3KTRML': 'Kinetic tremor', \
                      'NP3RTARU': 'Rest tremor', 'NP3RTALU': 'Rest tremor', 'NP3RTARL': 'Rest tremor', \
                      'NP3RTALL': 'Rest tremor', 'NP3RTALJ': 'Rest tremor', 'NP3RTCON': 'Rest tremor', \
                      'NP3RIGN': 'Rigidity', 'NP3RIGRU': 'Rigidity', 'NP3RIGLU': 'Rigidity', 'NP3RIGRL': 'Rigidity', \
                      'NP3RIGLL': 'Rigidity', \
                      'NP3FACXP': 'NP3FACXP', 'NP3RISNG': 'NP3RISNG', \
                      'NP3FTAPR': 'Finger tapping', 'NP3FTAPL': 'Finger tapping', \
                      'NP3PRSPL': 'Pronation-supination hands', 'NP3PRSPR': 'Pronation-supination hands', \
                      'NP3LGAGR': 'Leg agility', 'NP3LGAGL': 'Leg agility', \
                      'NP3HMOVR': 'Hand movements', 'NP3HMOVL': 'Hand movements', \
                      'NP3TTAPL': 'Toe tapping', 'NP3TTAPR': 'Toe tapping', \
                      'NP3PSTBL': 'Postural stability', \
                      'NP3BRADY': 'NP3BRADY', 'NP3POSTR': 'NP3POSTR', \
                      
                      'DFRTREMP': 'DF tremor', 'DFRTREMA': 'DF tremor', 'DFPATREM': 'DFPATREM', 'DFOTHTRM': 'DFOTHTRM', \
                      'DFRIGIDP': 'DF rigidity', 'DFRIGIDA': 'DF rigidity', 'DFAXRIG': 'DFAXRIG', 'DFUNIRIG': 'DFUNIRIG', \
                      'DFTONE': 'DFTONE', 'DFOTHRIG': 'DFOTHRIG', 'DFBRADYP': 'DF bradykinesia', \
                      'DFBRADYA': 'DF bradykinesia', 'DFAKINES': 'DFAKINES', 'DFBRPLUS': 'DFBRPLUS', \
                      'DFOTHABR': 'DFOTHABR', 'DFPGDIST': 'DFPGDIST', 'DFFALLS': 'DFFALLS', \
                      'DFGAIT': 'DFGAIT', 'DFFREEZ': 'DFFREEZ', 'DFOTHPG': 'DFOTHPG', 'DFDYSTON': 'DFDYSTON', \
                      'DFCHOREA': 'DFCHOREA', 'DFMYOCLO': 'DFMYOCLO', 'DFOTHHYP': 'DFOTHHYP', 'DFBULBAR': 'DFBULBAR'}
motor_spec_dict = dict()
for feat in motor_group_lookup.keys():
    feat_group = motor_group_lookup[feat]
    if feat_group not in motor_spec_dict.keys():
        motor_spec_dict[feat_group] = dict()
    if feat.startswith('DF'):
        motor_spec_dict[feat_group][feat] = ([1], True)
    else:
        motor_spec_dict[feat_group][feat] = (get_feat_threshs_to_try(pd_df[feat].values, 'above'), True)
    min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
spec_dict['Motor'] = motor_spec_dict

'''
Majority of cognitive features should be mapped to (below_thresh_percentiles, False) with 3 exceptions:
1. 'DFCOGNIT' should be mapped to ([1], True)
2. 'COGSTATE' should be mapped to ([1,2], True)
3. 'NP1COG' and 'DVT_SDM' should be mapped to (above_thresh_percentiles, True)
Only 1 grouping: 'MCAREC1','MCAREC2','MCAREC3','MCAREC4','MCAREC5' belong in 'MoCA recall' group
'''
recall_group = {'MCAREC1','MCAREC2','MCAREC3','MCAREC4','MCAREC5'}
cog_feats = {'COGSTATE', 'NP1COG', 'DVT_SDM', 'DFCOGNIT', 'MCAALTTM', 'MCACUBE', 'MCACLCKC', 'MCACLCKN', 'MCACLCKH', \
             'MCALION', 'MCARHINO', 'MCACAMEL', 'MCAFDS', 'MCABDS', 'MCAVIGIL', 'MCASER7', 'MCASNTNC', 'MCAVF', 'MCAABSTR', \
             'MCAREC1', 'MCAREC2', 'MCAREC3', 'MCAREC4', 'MCAREC5', 'MCADATE', 'MCAMONTH', 'MCAYR', 'MCADAY', 'MCAPLACE', \
             'MCACITY', 'HVLT_immed_recall', 'HVLT_discrim_recog', 'HVLT_retent', 'VLTANIM', 'VLTVEG', 'VLTFRUIT', \
             'COGDECLN', 'FNCDTCOG'}
cog_feats = cog_feats.union(set(['BJLOT' + str(idx) for idx in range(1, 31)]))
cog_feats = cog_feats.union(set(['LNS' + str(i) + letter for i in range(1,8) for letter in ['A', 'B', 'C']]))
cog_spec_dict = {'MoCA recall': dict()}
for feat in cog_feats:
    if feat in recall_group:
        cog_spec_dict['MoCA recall'][feat] = (get_feat_threshs_to_try(pd_df[feat].values, 'below'), False)
    elif feat in {'DFCOGNIT', 'COGDECLN', 'FNCDTCOG'}:
        cog_spec_dict[feat] = {feat: ([1], True)}
    elif feat == 'COGSTATE':
        cog_spec_dict[feat] = {feat: ([1,2], True)}
    elif feat == 'NP1COG' or feat == 'DVT_SDM':
        cog_spec_dict[feat] = {feat: (get_feat_threshs_to_try(pd_df[feat].values, 'above'), True)}
    else:
        cog_spec_dict[feat] = {feat: (get_feat_threshs_to_try(pd_df[feat].values, 'below'), False)}
    min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
spec_dict['Cognitive'] = cog_spec_dict

'''
For psychiatric, because individual questions go in different directions, direction is specified below.
Special cases:
1. Only NP1 questions are not binary
2. 4 groupings in GDS
'''
all_psychiatric_at_leasts = {'GDSSATIS': False, 'GDSGSPIR': False, 'GDSHAPPY': False, 'GDSALIVE': False, \
                             'GDSENRGY': False, 'GDSDROPD': True, 'GDSEMPTY': True, 'GDSBORED': True, 'GDSAFRAD': True, \
                             'GDSHLPLS': True, 'GDSHOME': True, 'GDSMEMRY': True, 'GDSWRTLS': True, 'GDSHOPLS': True, \
                             'GDSBETER': True, 'NP1DPRS': True, 'NP1APAT': True, 'CNTRLGMB': True, 'TMGAMBLE': True, \
                             'CNTRLSEX': True, 'TMSEX': True, 'CNTRLBUY': True, 'TMBUY': True, 'CNTRLEAT': True, \
                             'TMEAT': True, 'TMTORACT': True, 'TMTMTACT': True, 'TMTRWD': True, 'NP1ANXS': True, \
                             'NP1HALL': True, 'DFPSYCH': True}
psych_group_lookup = {'GDSSATIS': 'GDS satisfied/alive', 'GDSALIVE': 'GDS satisfied/alive', \
                      'GDSGSPIR': 'GDS spirited/energy', 'GDSENRGY':'GDS spirited/energy', \
                      'GDSDROPD': 'GDS dropped interests/empty', 'GDSEMPTY': 'GDS dropped interests/empty', \
                      'GDSHLPLS': 'GDS hopeless/helpless', 'GDSHOPLS': 'GDS hopeless/helpless', \
                      'STAIAD1': 'STAI calm', 'STAIAD27': 'STAI calm', \
                      'STAIAD2': 'STAI secure', 'STAIAD33': 'STAI secure', \
                      'STAIAD3': 'STAI tense', 'STAIAD40': 'STAI tense', \
                      'STAIAD7': 'STAI worry', 'STAIAD17': 'STAI worry', 'STAIAD29': 'STAI worry', \
                      'STAIAD8': 'STAI satisfied', 'STAIAD23': 'STAI satisfied', \
                      'STAIAD11': 'STAI self-confident', 'STAIAD32': 'STAI self-confident', \
                      'STAIAD12': 'STAI nervous', 'STAIAD22': 'STAI nervous', \
                      'STAIAD14': 'STAI indecisive', 'STAIAD34': 'STAI indecisive', \
                      'STAIAD19': 'STAI steady', 'STAIAD39': 'STAI steady', \
                      'STAIAD20': 'STAI pleasant', 'STAIAD21': 'STAI pleasant'}

anxiety_forward_qs = [3, 4, 6, 7, 9, 12, 13, 14, 17, 18, 22, 24, 25, 28, 29, 31, 32, 35, 37, 38, 40]
for idx in range(1,41):
    if idx in anxiety_forward_qs:
        all_psychiatric_at_leasts['STAIAD'+str(idx)] = True
    else:
        all_psychiatric_at_leasts['STAIAD'+str(idx)] = False
psych_spec_dict = dict()
for group in set(psych_group_lookup.values()):
    psych_spec_dict[group] = dict()
for feat in all_psychiatric_at_leasts.keys():
    if feat.startswith('NP1'):
        psych_spec_dict[feat] = {feat: (get_feat_threshs_to_try(pd_df[feat].values, 'above'), True)}
    elif feat in psych_group_lookup.keys():
        psych_spec_dict[psych_group_lookup[feat]][feat] = ([1], True)
    else:
        if all_psychiatric_at_leasts[feat]:
            psych_spec_dict[feat] = {feat: (get_feat_threshs_to_try(pd_df[feat].values, 'above'), True)}
        else:
            psych_spec_dict[feat] = {feat: (get_feat_threshs_to_try(pd_df[feat].values, 'below'), False)}
    min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
spec_dict['Psychiatric'] = psych_spec_dict
        
'''
No exceptions for imaging
'''
all_imaging_at_least = {'contralateral_caudate': False, 'ipsilateral_caudate': False, \
                        'contralateral_putamen': False, 'ipsilateral_putamen': False, \
                        'count_density_ratio_contralateral': True, 'count_density_ratio_ipsilateral': True, \
                        'asymmetry_index_putamen': True, 'asymmetry_index_caudate': True}
imaging_spec_dict = dict()
for feat in all_imaging_at_least.keys():
    if all_imaging_at_least[feat]:
        imaging_spec_dict[feat] = {feat: (get_feat_threshs_to_try(pd_df[feat].values, 'above'), True)}
    else:
        imaging_spec_dict[feat] = {feat: (get_feat_threshs_to_try(pd_df[feat].values, 'below'), False)}
    min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
spec_dict['Imaging'] = imaging_spec_dict

'''
Autonomic: groupings for SCOPA-AUT features. DF features are binary
'''
scopa_aut_groupings = {'SCOPA mouth': {'SCAU1', 'SCAU2', 'SCAU3'}, 'SCAU4': {'SCAU4'}, \
                       'SCOPA bowel': {'SCAU5', 'SCAU6', 'SCAU7', 'SCAU26A'}, \
                       'SCOPA urine': {'SCAU8', 'SCAU9', 'SCAU10', 'SCAU11', 'SCAU12', 'SCAU13', 'SCAU_catheter', 'SCAU26B'}, \
                       'SCOPA hypotension': {'SCAU14', 'SCAU15', 'SCAU16', 'SCAU26C'}, 'SCOPA sweat': {'SCAU17', 'SCAU18'}, \
                       'SCAU19': {'SCAU19'}, 'SCAU temp': {'SCAU20', 'SCAU21'}, \
                       'SCOPA sex': {'SCAU22', 'SCAU23', 'SCAU23A', 'SCAU24', 'SCAU25'}}
np1_feats = {'NP1CNST', 'NP1URIN', 'NP1LTHD', 'NP1FATG', 'NP1PAIN'}
df_feats = {'DFPSHYPO', 'DFSEXDYS', 'DFURDYS', 'DFBWLDYS'}
vital_groupings = {'Heart rate': {'HRSUP', 'HRSTND'}, 'Blood pressure': {'DIASTND', 'SYSSTND'}}
auto_spec_dict = dict()
for grouping in scopa_aut_groupings.keys():
    auto_spec_dict[grouping] = dict()
    for feat in scopa_aut_groupings[grouping]:
        auto_spec_dict[grouping][feat] = (get_feat_threshs_to_try(pd_df[feat].values, 'above'), True)
        min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
for feat in np1_feats:
    auto_spec_dict[feat] = {feat: (get_feat_threshs_to_try(pd_df[feat].values, 'above'), True)}
    min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
for feat in df_feats:
    auto_spec_dict[feat] = {feat: ([1], True)}
    min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
for grouping in vital_groupings.keys():
    auto_spec_dict[grouping] = dict()
    for feat in vital_groupings[grouping]:
        if grouping == 'Heart rate':
            auto_spec_dict[grouping][feat] = (get_feat_threshs_to_try(pd_df[feat].values, 'above'), True)
        else: # Blood pressure
            auto_spec_dict[grouping][feat] = (get_feat_threshs_to_try(pd_df[feat].values, 'below'), False)
        min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
spec_dict['Autonomic'] = auto_spec_dict
    
'''
Sleep:
Continuous: NP1, ESS
Binary: RBD
'''
cont_feats = {'NP1SLPN', 'NP1SLPD', 'ESS1', 'ESS2', 'ESS3', 'ESS4', 'ESS5', 'ESS6', 'ESS7', 'ESS8'}
binary_feats = {'DRMVIVID', 'DRMAGRAC', 'DRMNOCTB', 'SLPLMBMV', 'SLPINJUR', 'DRMVERBL', 'DRMFIGHT', 'DRMUMV', 'DRMOBJFL', \
                'MVAWAKEN', 'DRMREMEM', 'SLPDSTRB'}
sleep_spec_dict = dict()
for feat in cont_feats:
    sleep_spec_dict[feat] = {feat: (get_feat_threshs_to_try(pd_df[feat].values, 'above'), True)}
    min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
for feat in binary_feats:
    sleep_spec_dict[feat] = {feat: ([1], True)}
    min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
spec_dict['Sleep'] = sleep_spec_dict
        

'''
Lower values are worse for all CSF features.
'''
csf_feats = {'ABETA_log', 'TTAU_log', 'PTAU_log', 'ASYNU_log', 'PTAU_ABETA_ratio', 'TTAU_ABETA_ratio', 'PTAU_TTAU_ratio'}
csf_spec_dict = dict()
for feat in csf_feats:
    csf_spec_dict[feat] = {feat: (get_feat_threshs_to_try(pd_df[feat].values, 'below'), False)}
    min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
spec_dict['CSF'] = csf_spec_dict

'''
Neurological exam. Most map to 1 being abnormal. Hyper reflexes are abnormal with 1 or 2.
Hypo reflexes and flexor/extensor plantar reflexes are omitted because unclear which is worse.
'''
cranial_nerve_feats = {'CN1RSP','CN2RSP','CN346RSP','CN5RSP','CN7RSP','CN8RSP','CN910RSP','CN11RSP','CN12RSP'}
reflex_groupings = {'Muscle strength': {'MSRARSP','MSLARSP','MSRLRSP','MSLLRSP'}, \
                    'Coordination': {'COFNRRSP','COFNLRSP','COHSRRSP','COHSLRSP'}, \
                    'Sensation': {'SENRARSP','SENLARSP','SENRLRSP','SENLLRSP'}, \
                    'Hyper muscle stretch reflexes': {'RFLRARSP_hyper','RFLLARSP_hyper','RFLRLRSP_hyper','RFLLLRSP_hyper'}}
neuro_spec_dict = dict()
for feat in cranial_nerve_feats:
    neuro_spec_dict[feat] = {feat: ([1], True)}
    min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
for grouping in reflex_groupings.keys():
    neuro_spec_dict[grouping] = dict()
    for feat in reflex_groupings[grouping]:
        if feat.endswith('_hyper'):
            neuro_spec_dict[grouping][feat] = ([1,2], True)
        else:
            neuro_spec_dict[grouping][feat] = ([1], True)
        min_max_dict[feat] = (np.nanmin(pd_df[feat].values), np.nanmax(pd_df[feat].values))
spec_dict['Neurological exam'] = neuro_spec_dict
     
'''
TODO: Hematology. May need to get binary indicators for outside normal range. Be careful with units.
If getting binary indicators, will need to modify input to outcome calculator to also get binary indicators.
'''

'''
Checks on the specifications:
1. All features are in the human-readable dictionary.
2. Each feature appears in only 1 grouping.
3. Each grouping appears in only 1 category.
4. All features are in the min-max dictionary.
'''
groupings_so_far = set()
features_so_far = set()
for category in spec_dict.keys():
    cat_groupings = set(spec_dict[category].keys())
    if len(cat_groupings.intersection(groupings_so_far)) != 0:
        print('grouping duplicated')
        print(at_groupings.intersection(groupings_so_far))
    assert len(cat_groupings.intersection(groupings_so_far)) == 0
    groupings_so_far = groupings_so_far.union(cat_groupings)
    for grouping in cat_groupings:
        group_feats = set(spec_dict[category][grouping].keys())
        if len(group_feats.intersection(features_so_far)) != 0:
            print('features duplicated')
            print(group_feats.intersection(features_so_far))
        assert len(group_feats.intersection(features_so_far)) == 0
        features_so_far = features_so_far.union(group_feats)
        if not group_feats.issubset(set(human_readable_dict.keys())):
            print('features missing from human-readable dictionary')
            print(group_feats - set(human_readable_dict.keys()))
        assert group_feats.issubset(set(human_readable_dict.keys()))
        assert group_feats.issubset(set(min_max_dict.keys()))

'''
Store the specification dictionaries.
'''
spec_dir = 'survival_outcome_questions/'
if not os.path.isdir(spec_dir):
    os.makedirs(spec_dir)
with open(spec_dir + 'specs.pkl', 'w') as f:
    pickle.dump(spec_dict, f)
with open(spec_dir + 'human_readable_dict.pkl', 'w') as f:
    pickle.dump(human_readable_dict, f)
with open(spec_dir + 'min_max_dict.pkl', 'w') as f:
    pickle.dump(min_max_dict, f)