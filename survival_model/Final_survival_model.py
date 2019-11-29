import numpy as np, pandas as pd, pickle, matplotlib as mpl, os, sys, copy
mpl.use('agg')
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter, WeibullAFTFitter
np.random.seed(28033)
plt.rcParams.update({'font.size': 14})

def calc_maes(df, outcome, truncate_time):
    assert {outcome + '_T', outcome + '_E', outcome + '_T_pred'}.issubset(set(df.columns.values.tolist()))
    df[outcome + '_T_pred'] = np.where(df[outcome + '_T_pred'] > truncate_time, truncate_time, df[outcome + '_T_pred'])
    obs_df = df.loc[df[outcome + '_E']==1]
    cens_df = df.loc[df[outcome + '_E']==0]
    if len(obs_df) > 0:
        obs_mae = ((obs_df[outcome + '_T'] - obs_df[outcome + '_T_pred'])).abs().mean()
    else:
        obs_mae = 0
    if len(cens_df) > 0:
        cens_mae = np.mean(np.where(cens_df[outcome + '_T_pred'] < cens_df[outcome + '_T'], \
                           np.abs(cens_df[outcome + '_T_pred'] - cens_df[outcome + '_T']), 0))
    else:
        cens_mae = 0
    mae = (len(obs_df)*obs_mae + len(cens_df)*cens_mae)/len(df)
    return obs_mae, cens_mae, mae

def get_ci_mae_preds(model, df, df_patnos, model_type, outcome, truncate_time):
    assert len(df) == len(df_patnos)
    pred_df = model.predict_median(df)
    if model_type == 'Cox':
        pred_df.rename(columns={0.5: outcome + '_T_pred'}, inplace=True)
    else:
        pred_df.rename(columns={0: outcome + '_T_pred'}, inplace=True)
    pred_df[outcome + '_T'] = df[outcome + '_T']
    pred_df[outcome + '_E'] = df[outcome + '_E']
    ci = concordance_index(pred_df[outcome + '_T'], pred_df[outcome + '_T_pred'], pred_df[outcome + '_E'])
    _, _, mae = calc_maes(pred_df, outcome, truncate_time)
    pred_df_return = df_patnos.copy().reset_index(drop=True)
    pred_df_copy = pred_df[[outcome + '_T_pred']].reset_index(drop=True)
    pred_df_return[outcome + '_T_pred'] = pred_df_copy[outcome + '_T_pred']
    return ci, mae, pred_df_return

def main():
    '''
    Take name of parameter set as first parameter.
    '''
    param_err_msg = 'Expected name of covariate set as only parameter.'
    if len(sys.argv) != 2:
        print(param_err_msg)
        sys.exit()
    baseline_filepath = 'final_all_covariate_sets.pkl'
    with open(baseline_filepath, 'r') as f:
        baseline_feat_dicts = pickle.load(f)
    covariate_set_name = sys.argv[1]
    if covariate_set_name not in baseline_feat_dicts.keys():
        print(param_err_msg)
        print('Allowable options:')
        print(baseline_feat_dicts.keys())
        sys.exit()
    outcome_baseline_feats = baseline_feat_dicts[covariate_set_name]
    assert {'Motor','Autonomic','Cognitive','Psychiatric','Sleep','Standard'} == set(outcome_baseline_feats.keys())

    outcome_filepath = '../ppmi_survival_models/survival_outcome_subtotals_gdsfixed_using_CMEDTM/set_3.0_0.5_2019Jul08/' \
        + 'cohorts_time_event_dict.pkl'
    with open(outcome_filepath, 'r') as f:
        outcome_df = pickle.load(f)['PD']

    standard_baseline_feats = outcome_baseline_feats['Standard']
    del outcome_baseline_feats['Standard']
    output_dir = 'final_survival_models_' + covariate_set_name + '_2019Jul18/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open('final_human_readable_feat_dict.pkl', 'r') as f:
        human_readable_feat_dict = pickle.load(f)
    with open('final_test_patnos_dict.pkl', 'r') as f:
        test_patnos_dict = pickle.load(f)

    outcome_metrics = dict()
    # outcome: [Cox train CI avg, Cox train CI std, Cox train MAE avg, Cox train MAE std, Cox valid CI avg, Cox valid CI std, 
    # Cox valid MAE avg, Cox valid MAE std, Cox test CI avg, Cox test CI std, Cox test MAE avg, Cox test MAE std, 
    # Weibull train CI avg, Weibull train CI std, Weibull train MAE avg, Weibull train MAE std, 
    # Weibull valid CI avg, Weibull valid CI std, Weibull valid MAE avg, Weibull valid MAE std, 
    # Weibull test CI avg, Weibull test CI std, Weibull test MAE avg, Weibull test MAE std]
    outcomes = ['hybrid_requiremotor'] + outcome_baseline_feats.keys()
    print(outcomes)
    outcome_num_patnos = dict() # outcome to number of patients in train/valid
    for outcome in outcomes:
        baseline_filepath = 'final_survival_baseline_data.csv'
        baseline_df = pd.read_csv(baseline_filepath)
        if outcome == 'hybrid_requiremotor':
            selected_baseline_feats = copy.deepcopy(standard_baseline_feats)
            for outcome_ in outcome_baseline_feats.keys():
                selected_baseline_feats += outcome_baseline_feats[outcome_]
            if 'PhysExam_Psychiatric' in selected_baseline_feats:
                selected_baseline_feats.remove('PhysExam_Psychiatric') # some fold has 0 variance in training set
            if 'TMSEX' in selected_baseline_feats:
                selected_baseline_feats.remove('TMSEX') # same reason as above
            if 'TMTRWD' in selected_baseline_feats:
                selected_baseline_feats.remove('TMTRWD')
            if 'TMGAMBLE' in selected_baseline_feats:
                selected_baseline_feats.remove('TMGAMBLE')
        else:
            selected_baseline_feats = standard_baseline_feats + outcome_baseline_feats[outcome]
        selected_baseline_feats = list(set(selected_baseline_feats)) # in case duplicates were introduced
        selected_baseline_df = baseline_df[['PATNO']+selected_baseline_feats]
        df = outcome_df[['PATNO', outcome + '_T', outcome + '_E']].merge(selected_baseline_df, validate='one_to_one')
        df = df.dropna()
        df = df.loc[df[outcome + '_T']>0]
        for feat in selected_baseline_feats:
            df[feat] = (df[feat] - df[feat].min())/float(df[feat].max() - df[feat].min())
        all_patnos = set(df.PATNO.values.tolist())
        test_patnos = set(test_patnos_dict[outcome].tolist())
        '''
        missing_patnos = test_patnos.difference(all_patnos)
        if len(missing_patnos) > 0:
            print(outcome)
            missing_patnos_df = selected_baseline_df.loc[selected_baseline_df['PATNO'].isin(missing_patnos)]
            assert len(missing_patnos) == len(missing_patnos_df)
            for feat in selected_baseline_feats:
                if len(missing_patnos_df.dropna(subset=[feat])) != len(missing_patnos_df):
                    print(feat)
        continue
        '''
        assert test_patnos.issubset(all_patnos)
        train_valid_patnos = np.array(list(all_patnos.difference(test_patnos)))
        outcome_num_patnos[outcome] = len(train_valid_patnos)
        test_df = df.loc[df.PATNO.isin(test_patnos)]
        test_df_patnos = test_df[['PATNO']]
        del test_df['PATNO']
        cox_train_cis = []
        cox_train_maes = []
        cox_valid_cis = []
        cox_valid_maes = []
        cox_test_cis = []
        cox_test_maes = []
        cox_coefs = pd.DataFrame({'Feature': selected_baseline_feats})
        weibull_train_cis = []
        weibull_train_maes = []
        weibull_valid_cis = []
        weibull_valid_maes = []
        weibull_test_cis = []
        weibull_test_maes = []
        weibull_coefs = pd.DataFrame({'Feature': selected_baseline_feats})
        cox_penalizer_fig, cox_penalizer_ax = plt.subplots(nrows=2, ncols=4, figsize=(12,6))
        weibull_penalizer_fig, weibull_penalizer_ax = plt.subplots(nrows=2, ncols=4, figsize=(12,6))
        weibull_l1_ratio_fig, weibull_l1_ratio_ax = plt.subplots(nrows=2, ncols=4, figsize=(12,6))
        for fold_idx in range(4):
            valid_start_idx = int(fold_idx*.25*len(train_valid_patnos))
            valid_end_idx = int((fold_idx+1)*.25*len(train_valid_patnos))
            valid_patnos = train_valid_patnos[valid_start_idx:valid_end_idx]
            train_patnos = np.concatenate((train_valid_patnos[:valid_start_idx], train_valid_patnos[valid_end_idx:]))
            train_df = df.loc[df['PATNO'].isin(train_patnos)]
            train_df_patnos = train_df[['PATNO']]
            del train_df['PATNO']
            for col in train_df.columns:
                if train_df[col].std() < 0.05:
                    print(col + ': ' + str(train_df[col].mean()) + ', ' + str(train_df[col].std()))
            valid_df = df.loc[df['PATNO'].isin(valid_patnos)]
            valid_df_patnos = valid_df[['PATNO']]
            del valid_df['PATNO']
            cox_valid_ci_best = 0
            cox_valid_mae_best = float('inf')
            cox_penalizers = [0, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0, \
                              65.0, 80.0, 100.0, 120.0, 150.0, 200.0, 250.0, 300.0, 350., 400., 450., 500., \
                              600., 700., 800., 900., 1000., 1100., 1200., 1300., 1500., 1750., 2000., 2250., 2500., \
                              3000., 3500., 4000., 4500., 5000., 6000., 7000., 8000., 9000., 10000., 11000, 12500., 15000.]
            penalizer_train_cis = []
            penalizer_train_maes = []
            penalizer_valid_cis = []
            penalizer_valid_maes = []
            failed_to_conv_idxs = []
            cox_penalizer_best = None
            for penalizer_idx in range(len(cox_penalizers)):
                penalizer = cox_penalizers[penalizer_idx]
                cox_model = CoxPHFitter(penalizer=penalizer)
                converged = False
                itr = 0
                while not converged and itr < 10:
                    try:
                        print('Cox ' + outcome + ' ' + str(fold_idx) + ' ' + str(penalizer))
                        if itr != 0:
                            initial_weights = np.random.normal(scale=1.0, size = len(train_df.columns)-2)
                            cox_model.fit(train_df, duration_col = outcome + '_T', event_col = outcome + '_E', \
                                          initial_point = initial_weights)
                        else:
                            # default is all 0 initialization
                            cox_model.fit(train_df, duration_col = outcome + '_T', event_col = outcome + '_E')
                        converged = True
                    except:
                        itr += 1
                        continue
                if not converged:
                    penalizer_train_cis.append(0)
                    penalizer_train_maes.append(0)
                    penalizer_valid_cis.append(0)
                    penalizer_valid_maes.append(0)
                    failed_to_conv_idxs.append(penalizer_idx)
                    continue
                truncate_time = train_df[outcome + '_T'].max()
                cox_valid_ci, cox_valid_mae, cox_valid_pred_df = get_ci_mae_preds(cox_model, valid_df, valid_df_patnos, 'Cox', \
                                                                                  outcome, truncate_time)
                cox_train_ci, cox_train_mae, cox_train_pred_df = get_ci_mae_preds(cox_model, train_df, train_df_patnos, 'Cox', \
                                                                                  outcome, truncate_time)
                penalizer_train_cis.append(cox_train_ci)
                penalizer_train_maes.append(cox_train_mae)
                penalizer_valid_cis.append(cox_valid_ci)
                penalizer_valid_maes.append(cox_valid_mae)
                if cox_valid_ci > cox_valid_ci_best or (cox_valid_ci == cox_valid_ci_best and cox_valid_mae < cox_valid_mae_best):
                    cox_valid_ci_best = cox_valid_ci
                    cox_valid_mae_best = cox_valid_mae
                    cox_train_ci_best = cox_train_ci
                    cox_train_mae_best = cox_train_mae
                    cox_test_ci_best, cox_test_mae_best, cox_test_pred_df = get_ci_mae_preds(cox_model, test_df, test_df_patnos, \
                                                                                             'Cox', outcome, truncate_time)
                    cox_fold_coefs_best = cox_model.hazards_.copy()
                    cox_penalizer_best = penalizer
                    cox_test_pred_df.to_csv(output_dir + outcome + '_cox_fold' + str(fold_idx) + '_test_preds.csv', index=False)
                    cox_train_valid_pred_df = pd.concat([cox_train_pred_df, cox_valid_pred_df])
                    cox_train_valid_pred_df.to_csv(output_dir + outcome + '_cox_fold' + str(fold_idx) + '_train_valid_preds.csv', \
                                                   index=False)
            assert cox_valid_ci_best != 0
            cox_train_cis.append(cox_train_ci_best)
            cox_train_maes.append(cox_train_mae_best)
            cox_valid_cis.append(cox_valid_ci_best)
            cox_valid_maes.append(cox_valid_mae_best)
            cox_test_cis.append(cox_test_ci_best)
            cox_test_maes.append(cox_test_mae_best)
            cox_penalizers[0] = 1e-3
            if len(failed_to_conv_idxs) > 0:
                cox_penalizer_ax[0, fold_idx].scatter(np.array(cox_penalizers)[failed_to_conv_idxs],\
                                                      np.zeros(len(failed_to_conv_idxs)), c='b', marker='x')
                cox_penalizer_ax[0, fold_idx].scatter(np.array(cox_penalizers)[failed_to_conv_idxs], \
                                                      np.zeros(len(failed_to_conv_idxs)), c= 'r', marker='x')
                cox_penalizer_ax[1, fold_idx].scatter(np.array(cox_penalizers)[failed_to_conv_idxs], \
                                                      np.zeros(len(failed_to_conv_idxs)), c= 'b', marker='x')
                cox_penalizer_ax[1, fold_idx].scatter(np.array(cox_penalizers)[failed_to_conv_idxs], \
                                                      np.zeros(len(failed_to_conv_idxs)), c= 'r', marker='x')
                for idx in failed_to_conv_idxs[::-1]:
                    del penalizer_train_cis[idx]
                    del penalizer_train_maes[idx]
                    del penalizer_valid_cis[idx]
                    del penalizer_valid_maes[idx]
                    del cox_penalizers[idx]
            cox_penalizer_ax[0, fold_idx].plot(cox_penalizers, penalizer_train_cis, 'b', label='train')
            cox_penalizer_ax[0, fold_idx].plot(cox_penalizers, penalizer_valid_cis, 'r', label='valid')
            if cox_penalizer_best == 0:
                cox_penalizer_best_scatter_pt = 1e-3
            else:
                cox_penalizer_best_scatter_pt = cox_penalizer_best
            cox_penalizer_ax[0, fold_idx].scatter([cox_penalizer_best_scatter_pt], [cox_train_ci_best], c='b', marker='o')
            cox_penalizer_ax[0, fold_idx].scatter([cox_penalizer_best_scatter_pt], [cox_valid_ci_best], c='r', marker='o')
            cox_penalizer_ax[1, fold_idx].plot(cox_penalizers, penalizer_train_maes, 'b', label='train')
            cox_penalizer_ax[1, fold_idx].plot(cox_penalizers, penalizer_valid_maes, 'r', label='valid')
            cox_penalizer_ax[1, fold_idx].scatter([cox_penalizer_best_scatter_pt], [cox_train_mae_best], c='b', marker='o')
            cox_penalizer_ax[1, fold_idx].scatter([cox_penalizer_best_scatter_pt], [cox_valid_mae_best], c='r', marker='o')
            cox_penalizer_ax[0, fold_idx].set_xscale('log')
            cox_penalizer_ax[1, fold_idx].set_xscale('log')
            cox_penalizer_ax[0, fold_idx].set_xlabel('Penalizer')
            cox_penalizer_ax[1, fold_idx].set_xlabel('Penalizer')
            cox_penalizer_ax[0, fold_idx].set_ylabel('CI')
            cox_penalizer_ax[1, fold_idx].set_ylabel('MAE')
            cox_penalizer_ax[0, fold_idx].set_title('Fold ' + str(fold_idx))
            cox_penalizer_ax[1, fold_idx].set_title('Fold ' + str(fold_idx))
            cox_fold_coefs_best = cox_fold_coefs_best.reset_index()
            cox_fold_coefs_best.rename(columns={'index': 'Feature', 0: 'coef_fold' + str(fold_idx)}, inplace=True)
            cox_coefs = cox_coefs.merge(cox_fold_coefs_best, on=['Feature'], validate='one_to_one', suffixes=(False, False))

            weibull_valid_ci_best = 0
            weibull_valid_mae_best = float('inf')
            weibull_penalizers = [0, 0.01, 0.05, 0.1, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 10.0, 15.0, \
                                  20.0, 25.0, 35.0, 50.0, 65.0, 80.0, 100.0]
            weibull_l1_ratios = [0, 0.5, 1]
            penalizer_train_cis_dict = dict() # store all to plot corresponding to chosen l1 ratio
            penalizer_train_maes_dict = dict()
            penalizer_valid_cis_dict = dict()
            penalizer_valid_maes_dict = dict()
            penalizer_failed_to_conv_idxs = dict()
            for l1_ratio in weibull_l1_ratios:
                penalizer_train_cis_dict[l1_ratio] = []
                penalizer_train_maes_dict[l1_ratio] = []
                penalizer_valid_cis_dict[l1_ratio] = []
                penalizer_valid_maes_dict[l1_ratio] = []
                penalizer_failed_to_conv_idxs[l1_ratio] = []
            l1_ratio_train_cis_dict = dict()
            l1_ratio_train_maes_dict = dict()
            l1_ratio_valid_cis_dict = dict()
            l1_ratio_valid_maes_dict = dict()
            l1_ratio_failed_to_conv_idxs = dict()
            for penalizer in weibull_penalizers:
                l1_ratio_train_cis_dict[penalizer] = []
                l1_ratio_train_maes_dict[penalizer] = []
                l1_ratio_valid_cis_dict[penalizer] = []
                l1_ratio_valid_maes_dict[penalizer] = []
                l1_ratio_failed_to_conv_idxs[penalizer] = []
            best_penalizer = None
            best_l1_ratio = None
            for penalizer_idx in range(len(weibull_penalizers)):
                for l1_ratio_idx in range(len(weibull_l1_ratios)):
                    penalizer = weibull_penalizers[penalizer_idx]
                    l1_ratio = weibull_l1_ratios[l1_ratio_idx]
                    weibull_model = WeibullAFTFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                    try:
                        print('Weibull ' + outcome + ' ' + str(fold_idx) + ' ' + str(penalizer) + ' ' + str(l1_ratio))
                        weibull_model.fit(train_df, duration_col = outcome + '_T', event_col = outcome + '_E')
                    except:
                        penalizer_train_cis_dict[l1_ratio].append(0)
                        penalizer_train_maes_dict[l1_ratio].append(0)
                        penalizer_valid_cis_dict[l1_ratio].append(0)
                        penalizer_valid_maes_dict[l1_ratio].append(0)
                        penalizer_failed_to_conv_idxs[l1_ratio].append(penalizer_idx)
                        l1_ratio_train_cis_dict[penalizer].append(0)
                        l1_ratio_train_maes_dict[penalizer].append(0)
                        l1_ratio_valid_cis_dict[penalizer].append(0)
                        l1_ratio_valid_maes_dict[penalizer].append(0)
                        l1_ratio_failed_to_conv_idxs[penalizer].append(l1_ratio_idx)
                        continue
                    truncate_time = train_df[outcome + '_T'].max()
                    weibull_valid_ci, weibull_valid_mae, weibull_valid_pred_df \
                        = get_ci_mae_preds(weibull_model, valid_df, valid_df_patnos, 'Weibull', outcome, truncate_time)
                    weibull_train_ci, weibull_train_mae, weibull_train_pred_df \
                        = get_ci_mae_preds(weibull_model, train_df, train_df_patnos, 'Weibull', outcome, truncate_time)
                    penalizer_train_cis_dict[l1_ratio].append(weibull_train_ci)
                    penalizer_train_maes_dict[l1_ratio].append(weibull_train_mae)
                    penalizer_valid_cis_dict[l1_ratio].append(weibull_valid_ci)
                    penalizer_valid_maes_dict[l1_ratio].append(weibull_valid_mae)
                    l1_ratio_train_cis_dict[penalizer].append(weibull_train_ci)
                    l1_ratio_train_maes_dict[penalizer].append(weibull_train_mae)
                    l1_ratio_valid_cis_dict[penalizer].append(weibull_valid_ci)
                    l1_ratio_valid_maes_dict[penalizer].append(weibull_valid_mae)
                    if weibull_valid_ci > weibull_valid_ci_best \
                        or (weibull_valid_ci == weibull_valid_ci_best and weibull_valid_mae < weibull_valid_mae_best):
                        weibull_valid_ci_best = weibull_valid_ci
                        weibull_valid_mae_best = weibull_valid_mae
                        weibull_train_ci_best = weibull_train_ci
                        weibull_train_mae_best = weibull_train_mae
                        weibull_test_ci_best, weibull_test_mae_best, weibull_test_pred_df \
                            = get_ci_mae_preds(weibull_model, test_df, test_df_patnos, 'Weibull', outcome, truncate_time)
                        best_penalizer = penalizer
                        best_l1_ratio = l1_ratio
                        weibull_fold_coefs_best = weibull_model.params_.copy()
                        weibull_test_pred_df.to_csv(output_dir + outcome + '_weibull_fold' + str(fold_idx) + '_test_preds.csv', \
                                                    index=False)
                        weibull_train_valid_pred_df = pd.concat([weibull_train_pred_df, weibull_valid_pred_df])
                        weibull_train_valid_pred_df.to_csv(output_dir + outcome + '_weibull_fold' + str(fold_idx) \
                                                           + '_train_valid_preds.csv', index=False)
            assert weibull_valid_ci_best != 0
            weibull_train_cis.append(weibull_train_ci_best)
            weibull_train_maes.append(weibull_train_mae_best)
            weibull_valid_cis.append(weibull_valid_ci_best)
            weibull_valid_maes.append(weibull_valid_mae_best)
            weibull_test_cis.append(weibull_test_ci_best)
            weibull_test_maes.append(weibull_test_mae_best)
            weibull_penalizers[0] = 1e-3
            if len(penalizer_failed_to_conv_idxs[best_l1_ratio]) > 0:
                weibull_penalizer_ax[0, fold_idx].plot(np.array(weibull_penalizers)[penalizer_failed_to_conv_idxs[best_l1_ratio]], \
                                                       np.zeros(len(penalizer_failed_to_conv_idxs[best_l1_ratio])), c='b', \
                                                       marker='x')
                weibull_penalizer_ax[0, fold_idx].plot(np.array(weibull_penalizers)[penalizer_failed_to_conv_idxs[best_l1_ratio]], \
                                                       np.zeros(len(penalizer_failed_to_conv_idxs[best_l1_ratio])), c='r', \
                                                       marker='x')
                weibull_penalizer_ax[1, fold_idx].plot(np.array(weibull_penalizers)[penalizer_failed_to_conv_idxs[best_l1_ratio]], \
                                                       np.zeros(len(penalizer_failed_to_conv_idxs[best_l1_ratio])), c='b', \
                                                       marker='x')
                weibull_penalizer_ax[1, fold_idx].plot(np.array(weibull_penalizers)[penalizer_failed_to_conv_idxs[best_l1_ratio]], \
                                                       np.zeros(len(penalizer_failed_to_conv_idxs[best_l1_ratio])), c='r', \
                                                       marker='x')
                for idx in penalizer_failed_to_conv_idxs[best_l1_ratio][::-1]:
                    del weibull_penalizers[idx]
                    del penalizer_train_cis_dict[best_l1_ratio][idx]
                    del penalizer_valid_cis_dict[best_l1_ratio][idx]
                    del penalizer_train_maes_dict[best_l1_ratio][idx]
                    del penalizer_valid_maes_dict[best_l1_ratio][idx]
            weibull_penalizer_ax[0, fold_idx].plot(weibull_penalizers, penalizer_train_cis_dict[best_l1_ratio], 'b', label='train')
            weibull_penalizer_ax[0, fold_idx].plot(weibull_penalizers, penalizer_valid_cis_dict[best_l1_ratio], 'r', label='valid')
            if best_penalizer == 0:
                best_penalizer_scatter_pt = 1e-3
            else:
                best_penalizer_scatter_pt = best_penalizer
            weibull_penalizer_ax[0, fold_idx].scatter([best_penalizer_scatter_pt], [weibull_train_ci_best], c='b', marker='o')
            weibull_penalizer_ax[0, fold_idx].scatter([best_penalizer_scatter_pt], [weibull_valid_ci_best], c='r', marker='o')
            weibull_penalizer_ax[1, fold_idx].plot(weibull_penalizers, penalizer_train_maes_dict[best_l1_ratio], 'b', label='train')
            weibull_penalizer_ax[1, fold_idx].plot(weibull_penalizers, penalizer_valid_maes_dict[best_l1_ratio], 'r', label='valid')
            weibull_penalizer_ax[1, fold_idx].scatter([best_penalizer_scatter_pt], [weibull_train_mae_best], c='b', marker='o')
            weibull_penalizer_ax[1, fold_idx].scatter([best_penalizer_scatter_pt], [weibull_valid_mae_best], c='r', marker='o')
            weibull_penalizer_ax[0, fold_idx].set_xscale('log')
            weibull_penalizer_ax[1, fold_idx].set_xscale('log')
            weibull_penalizer_ax[0, fold_idx].set_xlabel('Penalizer')
            weibull_penalizer_ax[1, fold_idx].set_xlabel('Penalizer')
            weibull_penalizer_ax[0, fold_idx].set_ylabel('CI')
            weibull_penalizer_ax[1, fold_idx].set_ylabel('MAE')
            weibull_penalizer_ax[0, fold_idx].set_title('Fold ' + str(fold_idx))
            weibull_penalizer_ax[1, fold_idx].set_title('Fold ' + str(fold_idx))
            if len(l1_ratio_failed_to_conv_idxs[best_penalizer]) > 0:
                weibull_l1_ratio_ax[0, fold_idx].scatter(np.array(weibull_l1_ratios)[l1_ratio_failed_to_conv_idxs[best_penalizer]], \
                                                         np.zeros(len(l1_ratio_failed_to_conv_idxs[best_penalizer])), c='b', \
                                                         marker='x')
                weibull_l1_ratio_ax[0, fold_idx].scatter(np.array(weibull_l1_ratios)[l1_ratio_failed_to_conv_idxs[best_penalizer]], \
                                                         np.zeros(len(l1_ratio_failed_to_conv_idxs[best_penalizer])), c='r', \
                                                         marker='x')
                weibull_l1_ratio_ax[1, fold_idx].scatter(np.array(weibull_l1_ratios)[l1_ratio_failed_to_conv_idxs[best_penalizer]], \
                                                         np.zeros(len(l1_ratio_failed_to_conv_idxs[best_penalizer])), c='b', \
                                                         marker='x')
                weibull_l1_ratio_ax[1, fold_idx].scatter(np.array(weibull_l1_ratios)[l1_ratio_failed_to_conv_idxs[best_penalizer]], \
                                                         np.zeros(len(l1_ratio_failed_to_conv_idxs[best_penalizer])), c='r', \
                                                         marker='x')
                for idx in l1_ratio_failed_to_conv_idxs[best_penalizer][::-1]:
                    del weibull_l1_ratios[idx]
                    del l1_ratio_train_cis_dict[best_penalizer][idx]
                    del l1_ratio_valid_cis_dict[best_penalizer][idx]
                    del l1_ratio_train_maes_dict[best_penalizer][idx]
                    del l1_ratio_valid_maes_dict[best_penalizer][idx]
            weibull_l1_ratio_ax[0, fold_idx].plot(weibull_l1_ratios, l1_ratio_train_cis_dict[best_penalizer], 'b', label='train')
            weibull_l1_ratio_ax[0, fold_idx].plot(weibull_l1_ratios, l1_ratio_valid_cis_dict[best_penalizer], 'r', label='valid')
            weibull_l1_ratio_ax[0, fold_idx].scatter([best_l1_ratio], [weibull_train_ci_best], c='b', marker='o')
            weibull_l1_ratio_ax[0, fold_idx].scatter([best_l1_ratio], [weibull_valid_ci_best], c='r', marker='o')
            weibull_l1_ratio_ax[1, fold_idx].plot(weibull_l1_ratios, l1_ratio_train_maes_dict[best_penalizer], 'b', label='train')
            weibull_l1_ratio_ax[1, fold_idx].plot(weibull_l1_ratios, l1_ratio_valid_maes_dict[best_penalizer], 'r', label='valid')
            weibull_l1_ratio_ax[1, fold_idx].scatter([best_l1_ratio], [weibull_train_mae_best], c='b', marker='o')
            weibull_l1_ratio_ax[1, fold_idx].scatter([best_l1_ratio], [weibull_valid_mae_best], c='r', marker='o')
            weibull_l1_ratio_ax[0, fold_idx].set_xlabel('L1 ratio')
            weibull_l1_ratio_ax[1, fold_idx].set_xlabel('L1 ratio')
            weibull_l1_ratio_ax[0, fold_idx].set_ylabel('CI')
            weibull_l1_ratio_ax[1, fold_idx].set_ylabel('MAE')
            weibull_l1_ratio_ax[0, fold_idx].set_title('Fold ' + str(fold_idx))
            weibull_l1_ratio_ax[1, fold_idx].set_title('Fold ' + str(fold_idx))
            weibull_fold_coefs_best = weibull_fold_coefs_best.reset_index()
            weibull_fold_coefs_best = weibull_fold_coefs_best.loc[weibull_fold_coefs_best['level_1'] != '_intercept']
            del weibull_fold_coefs_best['level_0']
            weibull_fold_coefs_best.rename(columns={'level_1': 'Feature', 0:  'coef_fold' + str(fold_idx)}, \
                                           inplace=True)
            weibull_coefs = weibull_coefs.merge(weibull_fold_coefs_best, on=['Feature'], validate='one_to_one', \
                                                suffixes=(False, False))

        outcome_metrics[outcome] = [np.mean(np.array(cox_train_cis)), np.std(np.array(cox_train_cis)), \
                                    np.mean(np.array(cox_train_maes)), np.std(np.array(cox_train_maes)), \
                                    np.mean(np.array(cox_valid_cis)), np.std(np.array(cox_valid_cis)), \
                                    np.mean(np.array(cox_valid_maes)), np.std(np.array(cox_valid_maes)), \
                                    np.mean(np.array(cox_test_cis)), np.std(np.array(cox_test_cis)), \
                                    np.mean(np.array(cox_test_maes)), np.std(np.array(cox_test_maes)), \
                                    np.mean(np.array(weibull_train_cis)), np.std(np.array(weibull_train_cis)), \
                                    np.mean(np.array(weibull_train_maes)), np.std(np.array(weibull_train_maes)), \
                                    np.mean(np.array(weibull_valid_cis)), np.std(np.array(weibull_valid_cis)), \
                                    np.mean(np.array(weibull_valid_maes)), np.std(np.array(weibull_valid_maes)), \
                                    np.mean(np.array(weibull_test_cis)), np.std(np.array(weibull_test_cis)), \
                                    np.mean(np.array(weibull_test_maes)), np.std(np.array(weibull_test_maes))]
        cox_penalizer_ax[1, 3].legend()
        weibull_penalizer_ax[1, 3].legend()
        weibull_l1_ratio_ax[1, 3].legend()
        cox_penalizer_fig.tight_layout()
        cox_penalizer_fig.savefig(output_dir + outcome + '_cox_penalizer.pdf')
        weibull_penalizer_fig.tight_layout()
        weibull_penalizer_fig.savefig(output_dir + outcome + '_weibull_penalizer.pdf')
        weibull_l1_ratio_fig.tight_layout()
        weibull_l1_ratio_fig.savefig(output_dir + outcome + '_weibull_l1_ratio.pdf')
        cox_coefs['coef_mean'] = cox_coefs[['coef_fold' + str(fold_idx) for fold_idx in range(4)]].mean(axis=1)
        cox_coefs['coef_std'] = cox_coefs[['coef_fold' + str(fold_idx) for fold_idx in range(4)]].std(axis=1)
        cox_coefs = cox_coefs.sort_values(by='coef_mean')
        weibull_coefs['coef_mean'] = weibull_coefs[['coef_fold' + str(fold_idx) for fold_idx in range(4)]].mean(axis=1)
        weibull_coefs['coef_std'] = weibull_coefs[['coef_fold' + str(fold_idx) for fold_idx in range(4)]].std(axis=1)
        weibull_coefs = weibull_coefs.sort_values(by='coef_mean')
        cox_coefs.to_csv(output_dir + outcome + '_cox_coefs.csv', index=False)
        weibull_coefs.to_csv(output_dir + outcome + '_weibull_coefs.csv', index=False)
        cox_coef_fig, cox_coef_ax = plt.subplots(figsize=(9, len(cox_coefs)*.5+2))
        weibull_coef_fig, weibull_coef_ax = plt.subplots(figsize=(9, len(weibull_coefs)*.5+2))
        cox_coef_ax.errorbar(cox_coefs['coef_mean'].values, range(len(cox_coefs)), xerr=cox_coefs['coef_std'].values, fmt='o', \
                             capsize=3)
        cox_coef_ax.axvline(x=0, color='black')
        def reformat_coef_labels(labels):
            formatted_labels = []
            for label in labels:
                formatted_labels.append(human_readable_feat_dict[label])
            return formatted_labels
        cox_coef_ax.set_yticks(ticks=range(len(cox_coefs)))
        cox_coef_ax.set_yticklabels(reformat_coef_labels(cox_coefs['Feature'].values))
        cox_coef_ax.set_xlabel('Cox coefficients')
        cox_coef_fig.tight_layout()
        cox_coef_fig.savefig(output_dir + outcome + '_cox_coefs.pdf')
        weibull_coef_ax.errorbar(weibull_coefs['coef_mean'].values, range(len(weibull_coefs)), \
                                 xerr=weibull_coefs['coef_std'].values, fmt='o', capsize=3)
        weibull_coef_ax.axvline(x=0, color='black')
        weibull_coef_ax.set_yticks(ticks=range(len(weibull_coefs)))
        weibull_coef_ax.set_yticklabels(reformat_coef_labels(weibull_coefs['Feature'].values))
        weibull_coef_ax.set_xlabel('Weibull coefficients')
        weibull_coef_fig.tight_layout()
        weibull_coef_fig.savefig(output_dir + outcome + '_weibull_coefs.pdf')

    with open(output_dir + 'survival_metrics.pkl', 'w') as f:
        pickle.dump(outcome_metrics, f)
    # Cox: train: outcome: CI mean (CI std), MAE mean (MAE std), then valid:, then test:; then Weibull:
    metrics_str = 'Cox train:\n'
    outcomes = outcome_metrics.keys()
    for outcome in outcomes:
        metrics_str += outcome + ': {0:.4f}'.format(outcome_metrics[outcome][0]) \
            + ' ({0:.4f}), '.format(outcome_metrics[outcome][1]) + '{0:.4f}'.format(outcome_metrics[outcome][2]) \
            + ' ({0:.4f})\n'.format(outcome_metrics[outcome][3])
    metrics_str += 'Cox valid:\n'
    for outcome in outcomes:
        metrics_str += outcome + ': {0:.4f}'.format(outcome_metrics[outcome][4]) \
            + ' ({0:.4f}), '.format(outcome_metrics[outcome][5]) + '{0:.4f}'.format(outcome_metrics[outcome][6]) \
            + ' ({0:.4f})\n'.format(outcome_metrics[outcome][7])
    metrics_str += 'Cox test:\n'
    for outcome in outcomes:
        metrics_str += outcome + ': {0:.4f}'.format(outcome_metrics[outcome][8]) \
            + ' ({0:.4f}), '.format(outcome_metrics[outcome][9]) + '{0:.4f}'.format(outcome_metrics[outcome][10]) \
            + ' ({0:.4f})\n'.format(outcome_metrics[outcome][11])
    metrics_str += 'Weibull train:\n'
    for outcome in outcomes:
        metrics_str += outcome + ': {0:.4f}'.format(outcome_metrics[outcome][12]) \
            + ' ({0:.4f}), '.format(outcome_metrics[outcome][13]) + '{0:.4f}'.format(outcome_metrics[outcome][14]) \
            + ' ({0:.4f})\n'.format(outcome_metrics[outcome][15])
    metrics_str += 'Weibull valid:\n'
    for outcome in outcomes:
        metrics_str += outcome + ': {0:.4f}'.format(outcome_metrics[outcome][16]) \
            + ' ({0:.4f}), '.format(outcome_metrics[outcome][17]) + '{0:.4f}'.format(outcome_metrics[outcome][18]) \
            + ' ({0:.4f})\n'.format(outcome_metrics[outcome][19])
    metrics_str += 'Weibull test:\n'
    for outcome in outcomes:
        metrics_str += outcome + ': {0:.4f}'.format(outcome_metrics[outcome][20]) \
            + ' ({0:.4f}), '.format(outcome_metrics[outcome][21]) + '{0:.4f}'.format(outcome_metrics[outcome][22]) \
            + ' ({0:.4f})\n'.format(outcome_metrics[outcome][23])
    with open(output_dir + 'survival_metrics.txt', 'w') as f:
        f.write(metrics_str)

    with open(output_dir + 'sample_sizes.pkl', 'w') as f:
        pickle.dump(outcome_num_patnos, f)
    sample_sizes_str = ''
    for outcome in outcome_num_patnos.keys():
        sample_sizes_str += outcome + ': ' + str(outcome_num_patnos[outcome]) + '\n'
    with open(output_dir + 'sample_sizes.txt', 'w') as f:
        f.write(sample_sizes_str)
        
if __name__ == '__main__':
    main()