import pandas as pd, numpy as np, pickle, os, sys

if len(sys.argv) != 3:
    print('expecting path to ppmi directory and download date in format YYYYMmmDD as 1st and 2nd parameters')
    sys.exit()

ppmi_dir = sys.argv[1]
download_date = sys.argv[2]
if not ppmi_dir.endswith('/'):
    ppmi_dir += '/'
pipeline_dir = ppmi_dir + 'pipeline_output_asof_' + download_date + '_using_CMEDTM/'
if not os.path.isdir(pipeline_dir):
    print(pipeline_dir + ' as specified by input parameters does not exist')
    sys.exit()
aggregate_dir = ppmi_dir + 'visit_feature_inputs_asof_' + download_date + '_using_CMEDTM/'
if not os.path.isdir(aggregate_dir):
    os.makedirs(aggregate_dir)
agg_stat_df = pd.read_csv(pipeline_dir + 'agg_stat.csv')
cohorts = agg_stat_df.columns[1:]
files_in_each_cohort = ['_screening.csv', '_baseline.csv', '_totals_across_time.csv', '_questions_across_time.csv', \
                        '_other_across_time.csv']
# use pickle file to get binary columns
with open(pipeline_dir + 'binary_cols.pkl', 'r') as f:
    binary_cols = pickle.load(f)
with open(aggregate_dir + 'binary_cols.pkl', 'w') as f:
    pickle.dump(binary_cols, f)

standard3_cols = ['PATNO','EVENT_ID','INFODT']
time_cols = ['INFODT_DIS_DUR', 'INFODT_TIME_SINCE_ENROLL', 'EVENT_ID_DUR', 'DIS_DUR_BY_CONSENTDT']
    
def output_baseline_features(df):
    output_str = ''
    agg_stats = []
    agg_stat_names = []
    for feat in df.columns:
        if feat in standard3_cols or feat in time_cols:
            continue
        if feat == 'CNO':
            num_cnos = df.CNO.nunique()
            agg_stats.append(num_cnos)
            agg_stat_names.append('CNO num sites')
            continue
        nonnan_feat_df = df.loc[~pd.isnull(df[feat])][['PATNO',feat]]
        #print(nonnan_feat_df.head())
        if feat in binary_cols:
            if len(nonnan_feat_df) == 0:
                binary_freq1 = 0.
                feat_num_patnos = 0
            else:
                feat_num_patnos = nonnan_feat_df.PATNO.nunique()
                feat_vals = nonnan_feat_df[feat].value_counts()
                binary_freq1 = 0.
                for feat_val in feat_vals.keys():
                    if feat_val != 0:
                        binary_freq1 += feat_val*feat_vals[feat_val]/float(len(nonnan_feat_df))
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

def output_changing_features(df):
    output_str = ''
    agg_stats = []
    agg_stat_names = []
    for feat in df.columns:
        if feat in standard3_cols or feat in time_cols:
            continue
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
                binary_freq1 = 0.
                for feat_val in feat_vals.keys():
                    if feat_val != 0:
                        binary_freq1 += feat_val*feat_vals[feat_val]/float(len(nonnan_feat_df))
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

first = True
for cohort in cohorts:
    print(cohort)
    # get last infodt tied to a visit in any of the files
    cohort_infodt_df = pd.read_csv(pipeline_dir + cohort + files_in_each_cohort[0])[standard3_cols + time_cols].dropna(subset=['EVENT_ID'])
    for file in files_in_each_cohort[1:]:
        file_df = pd.read_csv(pipeline_dir + cohort + file)[standard3_cols + time_cols].dropna(subset=['EVENT_ID','INFODT'])
        cohort_infodt_df = pd.concat([cohort_infodt_df, file_df])
    cohort_infodt_df['INFODT'] = pd.to_datetime(cohort_infodt_df['INFODT'])
    cohort_infodt_df = cohort_infodt_df.sort_values(by=['INFODT'])
    cohort_infodt_df = cohort_infodt_df.drop_duplicates(subset=['PATNO','EVENT_ID'], keep='last')
    cohort_infodt_df['PATNO'] = cohort_infodt_df['PATNO'].astype(int)
    cohort_infodt_df['EVENT_ID'] = cohort_infodt_df['EVENT_ID'].astype(str)
    num_patnos = cohort_infodt_df.PATNO.nunique()
    cohort_output_str = cohort + '\n' + 'num_patnos: ' + str(num_patnos) + '\n'
    cohort_agg_stats = [num_patnos]
    if first:
        cohort_agg_stat_names = ['num_patnos']
     
    # baseline and screening already 1 visit per line, just align date with other files
    for file_ending in files_in_each_cohort[:2]:
        cohort_file_df = pd.read_csv(pipeline_dir + cohort + file_ending)
        cohort_file_df['PATNO'] = cohort_file_df['PATNO'].astype(int)
        cohort_file_df['EVENT_ID'] = cohort_file_df['EVENT_ID'].astype(str)
        del cohort_file_df['INFODT']
        for col in time_cols:
            del cohort_file_df[col]
        cohort_file_df = cohort_file_df.merge(cohort_infodt_df, on=['PATNO', 'EVENT_ID'], how='left', validate='one_to_one')
        cohort_file_df.to_csv(aggregate_dir + cohort + file_ending, index=False)
        cohort_file_output_str, cohort_file_agg_stats, cohort_file_agg_stat_names = output_baseline_features(cohort_file_df)
        cohort_output_str += cohort_file_output_str
        cohort_agg_stats += cohort_file_agg_stats
        if first:
            cohort_agg_stat_names += cohort_file_agg_stat_names
        
    # aggregate features across time so 1 row per patient visit
    for file_ending in files_in_each_cohort[2:]:
        cohort_file_df = pd.read_csv(pipeline_dir + cohort + file_ending)
        for col in time_cols:
            del cohort_file_df[col]
        # for rows that have date but no visit number, assign them to visit immediately following date
        cohort_file_df_no_eventid_df = cohort_file_df.loc[pd.isnull(cohort_file_df['EVENT_ID'])].dropna(subset=['INFODT'])
        del cohort_file_df_no_eventid_df['EVENT_ID']
        cohort_file_df_no_eventid_df['INFODT'] = pd.to_datetime(cohort_file_df_no_eventid_df['INFODT'])
        cohort_file_df_no_eventid_df = cohort_file_df_no_eventid_df.sort_values(by=['INFODT'])
        cohort_file_df_no_eventid_df['PATNO'] = cohort_file_df_no_eventid_df['PATNO'].astype(int)
        cohort_infodt_df = cohort_infodt_df.dropna(subset=['INFODT'])
        cohort_file_df_no_eventid_df = pd.merge_asof(cohort_file_df_no_eventid_df, cohort_infodt_df, on=['INFODT'], by=['PATNO'], direction='forward')
        for col in time_cols:
            del cohort_file_df_no_eventid_df[col]
        cohort_file_df = cohort_file_df.dropna(subset=['EVENT_ID'])
        cohort_file_df = pd.concat([cohort_file_df, cohort_file_df_no_eventid_df])
        cohort_file_df = cohort_file_df.dropna(subset=['EVENT_ID'])
        del cohort_file_df['INFODT']
        agg_dict = dict()
        for col in cohort_file_df.columns:
            if col == 'PATNO' or col == 'EVENT_ID':
                continue
            col_agg_dict = dict()
            col_agg_dict[col] = np.nanmean
            agg_dict[col] = col_agg_dict
            cohort_file_df[col] = cohort_file_df[col].astype(np.float64)
        mean_df = cohort_file_df.groupby(by=['PATNO','EVENT_ID']).agg(agg_dict)
        mean_df.columns = mean_df.columns.droplevel(0)
        mean_df = mean_df.reset_index()
        mean_df['PATNO'] = mean_df['PATNO'].astype(int)
        mean_df['EVENT_ID'] = mean_df['EVENT_ID'].astype(str)
        mean_df = mean_df.merge(cohort_infodt_df, on=['PATNO', 'EVENT_ID'], how='left', validate='one_to_one')
        if len(mean_df.loc[pd.isnull(mean_df['EVENT_ID_DUR'])]) != 0:
            print(cohort + file_ending + ' has missing disease duration')
        mean_df.to_csv(aggregate_dir + cohort + file_ending, index=False)
        cohort_file_output_str, cohort_file_agg_stats, cohort_file_agg_stat_names = output_baseline_features(mean_df)
        cohort_output_str += cohort_file_output_str
        cohort_agg_stats += cohort_file_agg_stats
        if first:
            cohort_agg_stat_names += cohort_file_agg_stat_names
    if first:
        agg_stats_df = pd.DataFrame(cohort_agg_stat_names, columns=['agg_stats'])
        first = False
    agg_stats_df[cohort] = cohort_agg_stats
    with open(aggregate_dir + cohort + '_summary.txt', 'w') as f:
        f.write(cohort_output_str)
    agg_stats_df.to_csv(aggregate_dir + 'agg_stats.csv', index=False)