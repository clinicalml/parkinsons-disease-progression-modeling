import numpy as np, pandas as pd, sys

if len(sys.argv) < 2:
    print('Pass in path to PPMI data directory as parameter.')
    sys.exit()
datadir = sys.argv[1]
pd_baseline_df = pd.read_csv(datadir + 'PD_baseline.csv')
pd_screening_df = pd.read_csv(datadir + 'PD_screening.csv')
pd_totals_df = pd.read_csv(datadir + 'PD_totals_across_time.csv')
pd_questions_df = pd.read_csv(datadir + 'PD_questions_across_time.csv')
pd_other_df = pd.read_csv(datadir + 'PD_other_across_time.csv')
pd_totals_baseline_df = pd_totals_df.loc[pd_totals_df['EVENT_ID']=='BL']
pd_totals_screening_df = pd_totals_df.loc[pd_totals_df['EVENT_ID']=='SC']
pd_questions_baseline_df = pd_questions_df.loc[pd_questions_df['EVENT_ID']=='BL']
pd_questions_screening_df = pd_questions_df.loc[pd_questions_df['EVENT_ID']=='SC']
pd_other_baseline_df = pd_other_df.loc[pd_other_df['EVENT_ID']=='BL']
pd_other_screening_df = pd_other_df.loc[pd_other_df['EVENT_ID']=='SC']
pd_baseline_df['FAMHIST'] = np.where(pd_baseline_df[['BIOMOMPD', 'BIODADPD', 'FULSIBPD', 'HAFSIBPD', 'MAGPARPD', \
                                                     'PAGPARPD', 'MATAUPD', 'PATAUPD', 'KIDSPD']].sum(axis=1) > 0, \
                                     1, 0)
desired_baseline_cols = ['MALE', 'RAWHITE', 'FAMHIST', 'EDUCYRS', 'RIGHT_HANDED', 'UPSIT', 'DIS_DUR_BY_CONSENTDT']
desired_pd_baseline_df = pd_baseline_df[['PATNO']+desired_baseline_cols]
genetic_cols = ['rs823118_T/T', 'rs823118_C/T',
       'rs823118_C/C', 'rs3910105_C/C', 'rs3910105_T/T', 'rs3910105_C/T',
       'rs356181_C/T', 'rs356181_T/T', 'rs356181_C/C', 'rs55785911_G/G',
       'rs55785911_A/G', 'rs55785911_A/A', 'rs2414739_A/A',
       'rs2414739_A/G', 'rs2414739_G/G', 'rs329648_C/C', 'rs329648_C/T',
       'rs329648_T/T', 'rs11724635_A/C', 'rs11724635_A/A',
       'rs11724635_C/C', 'rs17649553_C/C', 'rs17649553_C/T',
       'rs17649553_T/T', 'rs114138760_G/G', 'rs114138760_C/G',
       'ApoE Genotype_e3/e3', 'ApoE Genotype_e3/e2',
       'ApoE Genotype_e4/e3', 'ApoE Genotype_e2/e4',
       'ApoE Genotype_e4/e4', 'ApoE Genotype_e2/e2', 'rs11868035_G/G',
       'rs11868035_A/G', 'rs11868035_A/A', 'rs71628662_T/T',
       'rs71628662_C/T', 'rs118117788_C/C', 'rs118117788_C/T',
       'rs11158026_C/T', 'rs11158026_C/C', 'rs11158026_T/T',
       'rs34884217_T/T', 'rs34884217_G/T', 'rs34884217_G/G',
       'rs34311866_G/G', 'rs34311866_A/A', 'rs34311866_A/G',
       'rs199347_T/T', 'rs199347_C/T', 'rs199347_C/C', 'rs6430538_C/T',
       'rs6430538_C/C', 'rs6430538_T/T', 'rs34995376_LRRK2_p.R1441H_G/G',
       'rs11060180_G/G', 'rs11060180_A/A', 'rs11060180_A/G',
       'rs76763715_GBA_p.N370S_T/T', 'rs76763715_GBA_p.N370S_C/T',
       'rs12637471_G/G', 'rs12637471_A/G', 'rs12637471_A/A',
       'rs8192591_C/C', 'rs8192591_C/T', 'rs12456492_A/G',
       'rs12456492_A/A', 'rs12456492_G/G', 'rs14235_A/G', 'rs14235_A/A',
       'rs14235_G/G', 'rs35801418_LRRK2_p.Y1699C_A/A', 'rs591323_G/G',
       'rs591323_A/A', 'rs591323_A/G', 'rs6812193_C/C', 'rs6812193_C/T',
       'rs6812193_T/T', 'rs76904798_C/C', 'rs76904798_C/T',
       'rs76904798_T/T', 'rs34637584_LRRK2_p.G2019S_G/G',
       'rs34637584_LRRK2_p.G2019S_A/G', 'rs10797576_C/C',
       'rs10797576_C/T', 'rs10797576_T/T', 'rs115462410_C/C',
       'rs115462410_C/T', 'rs115462410_T/T', 'rs1955337_G/G',
       'rs1955337_G/T', 'rs1955337_T/T', 'rs35870237_LRRK2_p.I2020T_T/T',
       'rs421016_A/A', 'rs421016_A/G', 'rs76763715_T/T', 'rs76763715_C/T',
       'rs76763715_C/C', 'rs75548401_G/G', 'rs75548401_G/A',
       'rs2230288_C/C', 'rs2230288_C/T', 'rs104886460_C/C',
       'rs104886460_C/T', 'rs387906315_G/G', 'rs387906315_G/GC',
       'rs4653767_T/T', 'rs4653767_T/C', 'rs4653767_C/C',
       'rs34043159_T/T', 'rs34043159_T/C', 'rs34043159_C/C',
       'rs353116_C/C', 'rs353116_C/T', 'rs353116_T/T', 'rs4073221_T/T',
       'rs4073221_T/G', 'rs4073221_G/G', 'rs12497850_T/T',
       'rs12497850_T/G', 'rs12497850_G/G', 'rs143918452_G/G',
       'rs143918452_G/A', 'rs1803274_C/C', 'rs1803274_C/T',
       'rs1803274_T/T', 'rs104893877_C/C', 'rs104893877_C/T',
       'rs104893875_G/G', 'rs104893878_G/G', 'rs4444903_A/A',
       'rs4444903_A/G', 'rs4444903_G/G', 'rs121434567_C/C',
       'rs78738012_T/T', 'rs78738012_T/C', 'rs78738012_C/C',
       'rs2694528_A/A', 'rs2694528_A/C', 'rs2694528_C/C', 'rs9468199_G/G',
       'rs9468199_G/A', 'rs9468199_A/A', 'rs1293298_A/A', 'rs1293298_A/C',
       'rs1293298_C/C', 'rs2280104_C/C', 'rs2280104_C/T', 'rs2280104_T/T',
       'rs13294100_G/G', 'rs13294100_G/T', 'rs13294100_T/T',
       'rs10906923_A/A', 'rs10906923_A/C', 'rs10906923_C/C',
       'rs33939927_C/C', 'rs33939927_C/G', 'rs33939927_C/T',
       'rs33949390_G/G', 'rs33949390_G/C', 'rs35801418_A/A',
       'rs34637584_G/G', 'rs34637584_A/G', 'rs34637584_A/A',
       'rs34778348_G/G', 'rs34778348_G/A', 'rs8005172_C/C',
       'rs8005172_C/T', 'rs8005172_T/T', 'rs11343_G/G', 'rs11343_G/T',
       'rs11343_T/T', 'rs4784227_C/C', 'rs4784227_C/T', 'rs4784227_T/T',
       'rs737866_T/T', 'rs737866_T/C', 'rs737866_C/C', 'rs174674_G/G',
       'rs174674_G/A', 'rs174674_A/A', 'rs5993883_G/G', 'rs5993883_G/T',
       'rs5993883_T/T', 'rs740603_G/G', 'rs740603_G/A', 'rs740603_A/A',
       'rs165656_C/C', 'rs165656_C/G', 'rs165656_G/G', 'rs6269_A/A',
       'rs6269_A/G', 'rs6269_G/G', 'rs4633_T/T', 'rs4633_T/C',
       'rs4633_C/C', 'rs2239393_A/A', 'rs2239393_A/G', 'rs2239393_G/G',
       'rs4818_C/C', 'rs4818_C/G', 'rs4818_G/G', 'rs4680_G/G',
       'rs4680_G/A', 'rs4680_A/A', 'rs165599_A/A', 'rs165599_A/G',
       'rs165599_G/G']

genetic_cols_few_nan = []
for col in genetic_cols:
    if len(pd_screening_df.dropna(subset=[col])) > .9*len(pd_screening_df):
        genetic_cols_few_nan.append(col)
with open('genetic_feature_names.npy', 'w') as f:
    np.save(f, np.array(genetic_cols_few_nan))
genetic_df = pd_screening_df[['PATNO']+genetic_cols_few_nan].drop_duplicates(subset=['PATNO']).dropna()
# Run PCA on the genetic features
from sklearn.decomposition import PCA
genetic_pca = PCA(n_components=10)
genetic_components = genetic_pca.fit_transform(genetic_df[genetic_cols_few_nan].values)
genetic_components_df = pd.DataFrame(genetic_components, \
                                     columns=['Genetic PCA component ' + str(i) for i in range(10)])
genetic_components_df['PATNO'] = genetic_df['PATNO'].values
with open('genetic_pca_components.npy', 'w') as f:
    np.save(f, genetic_pca.components_)
with open('genetic_pca_explained_variance.npy', 'w') as f:
    np.save(f, genetic_pca.explained_variance_)
pd_totals_baseline_df['STAI'] = pd_totals_baseline_df[['STATE_ANXIETY','TRAIT_ANXIETY']].sum(axis=1)
desired_totals_screening_cols = ['NUPDRS1', 'NUPDRS2', 'NUPDRS3_untreated', 'MOCA']
desired_totals_baseline_cols = ['SCOPA-AUT', 'HVLT_discrim_recog',
       'STAI', 'HVLT_immed_recall', 'QUIP',
       'EPWORTH', 'GDSSHORT', 'HVLT_retent', 'BJLO', 
       'LNS', 'SEMANTIC_FLUENCY', 'REMSLEEP']
desired_pd_totals_baseline_df = pd_totals_baseline_df[['PATNO']+desired_totals_baseline_cols]
desired_pd_totals_screening_df = pd_totals_screening_df[['PATNO']+desired_totals_screening_cols]
desired_other_screening_cols = ['AGE', 'RIGHT_DOMSIDE', 'SYSSTND', 'SYSSUP', 'HRSTND', 'HRSUP', 'DIASUP', 'DIASTND', \
                                'TEMPC', 'TD_PIGD_untreated:tremor', 'TD_PIGD_untreated:posture']
desired_other_baseline_cols = ['WGTKG', 'HTCM', 'DVT_SDM', 'PTAU_ABETA_ratio', 'TTAU_ABETA_ratio', 'PTAU_TTAU_ratio', \
                               'PTAU_log', 'ABETA_log', 'ASYNU_log', 'CSF Hemoglobin']

datscan_cols = ['ipsilateral_putamen', 'ipsilateral_caudate', 'count_density_ratio_ipsilateral', 
                'count_density_ratio_contralateral', 'contralateral_putamen', 'contralateral_caudate', 
                'asymmetry_index_caudate', 'asymmetry_index_putamen']
datscan_df = pd_other_df[['PATNO','EVENT_ID_DUR','EVENT_ID']+datscan_cols].dropna().sort_values(by=['PATNO','EVENT_ID_DUR'])
first_datscan_df = datscan_df.drop_duplicates(subset=['PATNO'])
desired_other_screening_cols += datscan_cols
desired_pd_other_screening_df = pd_other_screening_df[['PATNO']+desired_other_screening_cols]
desired_pd_other_baseline_df = pd_other_baseline_df[['PATNO']+desired_other_baseline_cols]
dfs = [desired_pd_baseline_df, genetic_components_df, desired_pd_other_screening_df, desired_pd_other_baseline_df, \
       desired_pd_totals_baseline_df, desired_pd_totals_screening_df]
df = dfs[0]
for idx in range(1, len(dfs)):
    df = df.merge(dfs[idx], on=['PATNO'], how='inner', validate='one_to_one')
df = df.dropna()
df.to_csv('PD_selected_baseline.csv', index=False)