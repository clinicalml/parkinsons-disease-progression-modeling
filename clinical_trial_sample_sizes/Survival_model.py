import numpy as np, pandas as pd, pickle, matplotlib as mpl, os, sys, copy
mpl.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter, WeibullAFTFitter
np.random.seed(28033)
plt.rcParams.update({'font.size': 14})

def calc_maes(df, outcome, truncate_time):
    assert {outcome + '_T', outcome + '_E', outcome + '_T_pred'}.issubset(set(df.columns.values.tolist()))
    df[outcome + '_T_pred_truncated'] = np.where(df[outcome + '_T_pred'] > truncate_time, truncate_time, df[outcome + '_T_pred'])
    obs_df = df.loc[df[outcome + '_E']==1]
    cens_df = df.loc[df[outcome + '_E']==0]
    if len(obs_df) > 0:
        obs_mae = ((obs_df[outcome + '_T'] - obs_df[outcome + '_T_pred_truncated'])).abs().mean()
    else:
        obs_mae = 0
    if len(cens_df) > 0:
        cens_mae = np.mean(np.where(cens_df[outcome + '_T_pred_truncated'] < cens_df[outcome + '_T'], \
                           np.abs(cens_df[outcome + '_T_pred_truncated'] - cens_df[outcome + '_T']), 0))
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

def get_auroc_acc_prec_rec(true_binary_df, pred_df, outcome, num_years):
    prob_df = pd.DataFrame({'PATNO': pred_df['PATNO'].values})
    if len(pred_df.loc[pred_df[outcome + '_T_pred'] != float('inf')]) == 0:
        prob_df[outcome + '_T_prob'] = 0.5
    else:
        pred_max = pred_df[outcome + '_T_pred'].loc[pred_df[outcome + '_T_pred'] != float('inf')].max() + 0.05
        pred_df[outcome + '_T_pred_for_auroc'] = np.where(pred_df[outcome + '_T_pred'] == float('inf'), pred_max, \
                                                          pred_df[outcome + '_T_pred'])
        prob_df[outcome + '_T_prob'] = (pred_max - pred_df[outcome + '_T_pred_for_auroc'])/float(pred_max)
    binary_pred_df = pd.DataFrame({'PATNO': pred_df['PATNO'].values})
    binary_pred_df[outcome + '_T_binary_pred'] \
        = np.where(pred_df[outcome + '_T_pred'] > num_years, 0, 1)
    auroc = roc_auc_score(true_binary_df[outcome + '_E'].values, prob_df[outcome + '_T_prob'].values)
    acc = accuracy_score(true_binary_df[outcome + '_E'].values, binary_pred_df[outcome + '_T_binary_pred'].values)
    prec = precision_score(true_binary_df[outcome + '_E'].values, binary_pred_df[outcome + '_T_binary_pred'].values)
    rec = recall_score(true_binary_df[outcome + '_E'].values, binary_pred_df[outcome + '_T_binary_pred'].values)
    return auroc, acc, prec, rec

def main():
    '''
    Take name of parameter set as first parameter.
    '''
    param_msg = 'Expecting 1st parameter to be name of covariate set and 2nd parameter to be length of trial period in years.'
    if len(sys.argv) != 3:
        print(param_msg)
        sys.exit()
    num_years = sys.argv[2]
    if num_years not in {'2', '3'}:
        print(param_msg)
        print('Allowable options for 2nd parameter: 2 or 3')
        sys.exit()
    covariate_set_name = sys.argv[1]
    #with open('../finalized_outcome_survival_models/final_all_covariate_sets.pkl', 'rb') as f:
    #    all_covariate_sets = pickle.load(f, encoding='latin1')
    with open('finalized_covariate_sets_filter' + num_years + '.pkl', 'rb') as f:
        all_covariate_sets = pickle.load(f, encoding='latin1')
    if covariate_set_name not in all_covariate_sets.keys():
        print(param_msg)
        print('Allowable options:')
        print(baseline_feats_dict.keys())
        sys.exit()
    #with open('feats_to_remove_filter' + num_years + '.pkl', 'rb') as f:
    #    feats_to_remove = pickle.load(f, encoding='latin1')
    baseline_feats_dict = all_covariate_sets[covariate_set_name]
    with open('../finalized_outcome_survival_models/final_human_readable_feat_dict.pkl', 'rb') as f:
        human_readable_feat_dict = pickle.load(f, encoding='latin1')
    
    outcome_df = pd.read_csv('PD_outcomes_filter' + num_years + 'yrs.csv')

    output_dir = 'survival_model_' + num_years + '_years_' + covariate_set_name + '_2019Jul23_metricsfixed/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    surv_df_filtered = pd.read_csv('PD_outcomes_filter' + num_years + 'yrs.csv')
    with open('test_patnos_filter' + num_years + '.pkl', 'rb') as f:
        filtered_test_patnos = pickle.load(f, encoding='latin1')
    with open('valid_patnos_filter' + num_years + '.pkl', 'rb') as f:
        valid_patnos_dict = pickle.load(f, encoding='latin1')
    
    cox_outcome_metrics = dict()
    weibull_outcome_metrics = dict()
    outcomes = ['Autonomic', 'Cognitive', 'Psychiatric', 'Motor', 'Sleep', 'hybrid_requiremotor', 'NUPDRS23_45', 'MOCA_25', \
                'MSEADLG_79']
    metric_list = ['auroc', 'acc', 'prec', 'rec', 'ci', 'mae']
    metric_human_readable_list = ['AUROC', 'Accuracy', 'Precision', 'Recall', 'CI', 'MAE']
    outcome_num_patnos = dict() # outcome to number of patients in train/valid
    for outcome in outcomes:
        baseline_filepath = '../finalized_outcome_survival_models/final_survival_baseline_data.csv'
        baseline_df = pd.read_csv(baseline_filepath)
        '''
        if outcome == 'hybrid_requiremotor' or outcome == 'MSEADLG_79':
            selected_baseline_feats = list(baseline_feats_dict['Standard']) + list(baseline_feats_dict['Autonomic']) \
                + list(baseline_feats_dict['Cognitive']) + list(baseline_feats_dict['Psychiatric']) \
                + list(baseline_feats_dict['Motor']) + list(baseline_feats_dict['Sleep'])
        elif outcome == 'NUPDRS23_45':
            selected_baseline_feats = list(baseline_feats_dict['Standard']) + list(baseline_feats_dict['Motor'])
        elif outcome == 'MOCA_25':
            selected_baseline_feats = list(baseline_feats_dict['Standard']) + list(baseline_feats_dict['Cognitive'])
        else:
            selected_baseline_feats = list(baseline_feats_dict['Standard']) + list(baseline_feats_dict[outcome])
        selected_baseline_feats = list(set(selected_baseline_feats).difference(feats_to_remove[outcome]))
        '''
        selected_baseline_feats = list(baseline_feats_dict[outcome])
        selected_baseline_df = baseline_df[['PATNO']+selected_baseline_feats]
        df = outcome_df[['PATNO', outcome + '_T', outcome + '_E']].merge(selected_baseline_df, validate='one_to_one')
        df = df.dropna()
        df = df.loc[df[outcome + '_T']>0]
        all_patnos = set(df.PATNO.values.tolist())
        test_patnos = set(filtered_test_patnos[outcome].tolist())
        for feat in selected_baseline_feats:
            df[feat] = (df[feat] - df[feat].min())/float(df[feat].max() - df[feat].min())
        assert test_patnos.issubset(all_patnos)
        train_valid_patnos = np.array(list(all_patnos.difference(test_patnos)))
        outcome_num_patnos[outcome] = len(train_valid_patnos)
        test_df = df.loc[df.PATNO.isin(test_patnos)]
        test_df_patnos = test_df[['PATNO']]
        del test_df['PATNO']
        cox_train_metrics = {'auroc': [], 'acc': [], 'prec': [], 'rec': [], 'ci': [], 'mae': []}
        cox_valid_metrics = {'auroc': [], 'acc': [], 'prec': [], 'rec': [], 'ci': [], 'mae': []}
        cox_test_metrics = {'auroc': [], 'acc': [], 'prec': [], 'rec': [], 'ci': [], 'mae': []}
        cox_coefs = pd.DataFrame({'Feature': selected_baseline_feats})
        weibull_train_metrics = {'auroc': [], 'acc': [], 'prec': [], 'rec': [], 'ci': [], 'mae': []}
        weibull_valid_metrics = {'auroc': [], 'acc': [], 'prec': [], 'rec': [], 'ci': [], 'mae': []}
        weibull_test_metrics = {'auroc': [], 'acc': [], 'prec': [], 'rec': [], 'ci': [], 'mae': []}
        weibull_coefs = pd.DataFrame({'Feature': selected_baseline_feats})
        cox_penalizer_fig, cox_penalizer_ax = plt.subplots(nrows=6, ncols=4, figsize=(20,30))
        weibull_penalizer_fig, weibull_penalizer_ax = plt.subplots(nrows=6, ncols=4, figsize=(20,30))
        weibull_l1_ratio_fig, weibull_l1_ratio_ax = plt.subplots(nrows=6, ncols=4, figsize=(20,30))
        for fold_idx in range(4):
            valid_patnos = set(valid_patnos_dict[outcome][fold_idx].tolist())
            assert valid_patnos.issubset(set(train_valid_patnos.tolist()))
            train_patnos = set(train_valid_patnos.tolist()).difference(valid_patnos)
            train_df = df.loc[df['PATNO'].isin(train_patnos)]
            train_df_patnos = train_df[['PATNO']]
            del train_df['PATNO']
            valid_df = df.loc[df['PATNO'].isin(valid_patnos)]
            valid_df_patnos = valid_df[['PATNO']]
            del valid_df['PATNO']
            cox_penalizers = [0, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0, \
                              65.0, 80.0, 100.0, 120.0, 150.0, 200.0, 250.0, 300.0, 350., 400., 450., 500., \
                              600., 700., 800., 900., 1000., 1100., 1200., 1300., 1500., 1750., 2000., 2250., 2500., \
                              3000., 3500., 4000., 4500., 5000., 6000., 7000., 8000., 9000., 10000., 11000, 12500., 15000.]
            cox_vary_penalizer_train_metrics = {'auroc': [], 'acc': [], 'prec': [], 'rec': [], 'ci': [], 'mae': []}
            cox_vary_penalizer_valid_metrics = {'auroc': [], 'acc': [], 'prec': [], 'rec': [], 'ci': [], 'mae': []}
            cox_valid_best_metrics = {'auroc': 0., 'acc': 0., 'prec': 0., 'rec': 0., 'ci': 0., 'mae': float('inf')}
            cox_train_best_metrics = {'auroc': 0., 'acc': 0., 'prec': 0., 'rec': 0., 'ci': 0., 'mae': float('inf')}
            cox_test_best_metrics = {'auroc': 0., 'acc': 0., 'prec': 0., 'rec': 0., 'ci': 0., 'mae': float('inf')}
            cox_failed_to_conv_penalizers = []
            cox_penalizer_best = None
            for penalizer in cox_penalizers:
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
                    cox_failed_to_conv_penalizers.append(penalizer)
                    continue
                cox_valid_ci, cox_valid_mae, cox_valid_pred_df = get_ci_mae_preds(cox_model, valid_df, valid_df_patnos, 'Cox', \
                                                                                  outcome, float(num_years))
                cox_train_ci, cox_train_mae, cox_train_pred_df = get_ci_mae_preds(cox_model, train_df, train_df_patnos, 'Cox', \
                                                                                  outcome, float(num_years))
                cox_vary_penalizer_train_metrics['ci'].append(cox_train_ci)
                cox_vary_penalizer_train_metrics['mae'].append(cox_train_mae)
                cox_vary_penalizer_valid_metrics['ci'].append(cox_valid_ci)
                cox_vary_penalizer_valid_metrics['mae'].append(cox_valid_mae)
                cox_train_auroc, cox_train_acc, cox_train_prec, cox_train_rec \
                    = get_auroc_acc_prec_rec(train_df[[outcome + '_E']], cox_train_pred_df, outcome, float(num_years))
                cox_vary_penalizer_train_metrics['auroc'].append(cox_train_auroc)
                cox_vary_penalizer_train_metrics['acc'].append(cox_train_acc)
                cox_vary_penalizer_train_metrics['prec'].append(cox_train_prec)
                cox_vary_penalizer_train_metrics['rec'].append(cox_train_rec)
                cox_valid_auroc, cox_valid_acc, cox_valid_prec, cox_valid_rec \
                    = get_auroc_acc_prec_rec(valid_df[[outcome + '_E']], cox_valid_pred_df, outcome, float(num_years))
                cox_vary_penalizer_valid_metrics['auroc'].append(cox_valid_auroc)
                cox_vary_penalizer_valid_metrics['acc'].append(cox_valid_acc)
                cox_vary_penalizer_valid_metrics['prec'].append(cox_valid_prec)
                cox_vary_penalizer_valid_metrics['rec'].append(cox_valid_rec)
                if cox_valid_auroc > cox_valid_best_metrics['auroc'] \
                    or (cox_valid_auroc == cox_valid_best_metrics['auroc'] and cox_valid_acc > cox_valid_best_metrics['acc']) \
                    or (cox_valid_auroc == cox_valid_best_metrics['auroc'] and cox_valid_acc == cox_valid_best_metrics['acc'] \
                        and cox_valid_prec > cox_valid_best_metrics['prec']) \
                    or (cox_valid_auroc == cox_valid_best_metrics['auroc'] and cox_valid_acc == cox_valid_best_metrics['acc'] \
                        and cox_valid_prec == cox_valid_best_metrics['prec'] \
                        and cox_valid_rec > cox_valid_best_metrics['rec']) \
                    or (cox_valid_auroc == cox_valid_best_metrics['auroc'] and cox_valid_acc == cox_valid_best_metrics['acc'] \
                        and cox_valid_prec == cox_valid_best_metrics['prec'] \
                        and cox_valid_rec == cox_valid_best_metrics['rec'] \
                        and cox_valid_ci > cox_valid_best_metrics['ci']) \
                    or (cox_valid_auroc == cox_valid_best_metrics['auroc'] and cox_valid_acc == cox_valid_best_metrics['acc'] \
                        and cox_valid_prec == cox_valid_best_metrics['prec'] \
                        and cox_valid_rec == cox_valid_best_metrics['rec'] \
                        and cox_valid_ci == cox_valid_best_metrics['ci'] and cox_valid_mae < cox_valid_best_metrics['mae']):
                    cox_valid_best_metrics['auroc'] = cox_valid_auroc
                    cox_valid_best_metrics['acc'] = cox_valid_acc
                    cox_valid_best_metrics['prec'] = cox_valid_prec
                    cox_valid_best_metrics['rec'] = cox_valid_rec
                    cox_valid_best_metrics['ci'] = cox_valid_ci
                    cox_valid_best_metrics['mae'] = cox_valid_mae
                    cox_train_best_metrics['auroc'] = cox_train_auroc
                    cox_train_best_metrics['acc'] = cox_train_acc
                    cox_train_best_metrics['prec'] = cox_train_prec
                    cox_train_best_metrics['rec'] = cox_train_rec
                    cox_train_best_metrics['ci'] = cox_train_ci
                    cox_train_best_metrics['mae'] = cox_train_mae
                    cox_test_ci, cox_test_mae, cox_test_pred_df \
                        = get_ci_mae_preds(cox_model, test_df, test_df_patnos, 'Cox', outcome, float(num_years))
                    cox_test_auroc, cox_test_acc, cox_test_prec, cox_test_rec \
                        = get_auroc_acc_prec_rec(test_df[[outcome + '_E']], cox_test_pred_df, outcome, float(num_years))
                    cox_test_best_metrics['auroc'] = cox_test_auroc
                    cox_test_best_metrics['acc'] = cox_test_acc
                    cox_test_best_metrics['prec'] = cox_test_prec
                    cox_test_best_metrics['rec'] = cox_test_rec
                    cox_test_best_metrics['ci'] = cox_test_ci
                    cox_test_best_metrics['mae'] = cox_test_mae
                    cox_fold_coefs_best = cox_model.hazards_.copy()
                    cox_penalizer_best = penalizer
                    cox_test_pred_df.to_csv(output_dir + outcome + '_cox_fold' + str(fold_idx) + '_test_preds.csv', index=False)
                    cox_train_valid_pred_df = pd.concat([cox_train_pred_df, cox_valid_pred_df])
                    cox_train_valid_pred_df.to_csv(output_dir + outcome + '_cox_fold' + str(fold_idx) \
                                                   + '_train_valid_preds.csv', index=False)
            assert cox_valid_best_metrics['auroc'] != 0
            for metric in cox_train_best_metrics.keys():
                cox_train_metrics[metric].append(cox_train_best_metrics[metric])
                cox_valid_metrics[metric].append(cox_valid_best_metrics[metric])
                cox_test_metrics[metric].append(cox_test_best_metrics[metric])
            cox_penalizers[0] = 1e-3
            if len(cox_failed_to_conv_penalizers) > 0:
                if cox_failed_to_conv_penalizers[0] == 0:
                    cox_failed_to_conv_penalizers[0] = 1e-3
                for penalizer in cox_failed_to_conv_penalizers:
                    cox_penalizers.remove(penalizer)
                for metric_idx in range(len(metric_list)):
                    cox_penalizer_ax[metric_idx, fold_idx].scatter(cox_failed_to_conv_penalizers, \
                                                                   np.zeros(len(cox_failed_to_conv_penalizers)), \
                                                                   c='r', marker='x')
            if cox_penalizer_best == 0:
                cox_penalizer_best_scatter_pt = 1e-3
            else:
                cox_penalizer_best_scatter_pt = cox_penalizer_best
            for metric_idx in range(len(metric_list)):
                metric = metric_list[metric_idx]
                curr_ax = cox_penalizer_ax[metric_idx, fold_idx]
                curr_ax.plot(cox_penalizers, cox_vary_penalizer_train_metrics[metric], 'b', linestyle='--', label='train')
                curr_ax.plot(cox_penalizers, cox_vary_penalizer_valid_metrics[metric], 'r', label='valid')
                curr_ax.scatter([cox_penalizer_best_scatter_pt], [cox_train_best_metrics[metric]], c='b')
                curr_ax.scatter([cox_penalizer_best_scatter_pt], [cox_valid_best_metrics[metric]], c='r')
                curr_ax.set_xscale('log')
                curr_ax.set_xlabel('Penalizer')
                curr_ax.set_ylabel(metric_human_readable_list[metric_idx])
                curr_ax.set_title('Fold ' + str(fold_idx))
            cox_fold_coefs_best = cox_fold_coefs_best.reset_index()
            cox_fold_coefs_best.rename(columns={'index': 'Feature', 0: 'coef_fold' + str(fold_idx)}, inplace=True)
            cox_coefs = cox_coefs.merge(cox_fold_coefs_best, on=['Feature'], validate='one_to_one', suffixes=(False, False))

            weibull_valid_best_metrics = {'auroc': 0., 'acc': 0., 'prec': 0., 'rec': 0., 'ci': 0., 'mae': float('inf')}
            weibull_train_best_metrics = {'auroc': 0., 'acc': 0., 'prec': 0., 'rec': 0., 'ci': 0., 'mae': float('inf')}
            weibull_test_best_metrics = {'auroc': 0., 'acc': 0., 'prec': 0., 'rec': 0., 'ci': 0., 'mae': float('inf')}
            weibull_penalizers = [0, 0.01, 0.05, 0.1, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, \
                                  10.0, 15.0, 20.0, 25.0, 35.0, 50.0, 65.0, 80.0, 100.0]
            weibull_l1_ratios = [0, 0.5, 1]
            weibull_vary_penalizer_train_metrics = dict()
            weibull_vary_penalizer_valid_metrics = dict()
            weibull_failed_to_conv_penalizers = dict()
            for l1_ratio in weibull_l1_ratios:
                weibull_vary_penalizer_train_metrics[l1_ratio] = {'auroc': [], 'acc': [], 'prec': [], 'rec': [], 'ci': [], \
                                                                  'mae': []}
                weibull_vary_penalizer_valid_metrics[l1_ratio] = {'auroc': [], 'acc': [], 'prec': [], 'rec': [], 'ci': [], \
                                                                  'mae': []}
                weibull_failed_to_conv_penalizers[l1_ratio] = []
            weibull_vary_l1_ratio_train_metrics = dict()
            weibull_vary_l1_ratio_valid_metrics = dict()
            weibull_failed_to_conv_l1_ratios = dict()
            for penalizer in weibull_penalizers:
                weibull_vary_l1_ratio_train_metrics[penalizer] = {'auroc': [], 'acc': [], 'prec': [], 'rec': [], 'ci': [], \
                                                                  'mae': []}
                weibull_vary_l1_ratio_valid_metrics[penalizer] = {'auroc': [], 'acc': [], 'prec': [], 'rec': [], 'ci': [], \
                                                                  'mae': []}
                weibull_failed_to_conv_l1_ratios[penalizer] = []
            weibull_best_penalizer = None
            weibull_best_l1_ratio = None
            for penalizer in weibull_penalizers:
                for l1_ratio in weibull_l1_ratios:
                    weibull_model = WeibullAFTFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                    try:
                        print('Weibull ' + outcome + ' ' + str(fold_idx) + ' ' + str(penalizer) + ' ' + str(l1_ratio))
                        weibull_model.fit(train_df, duration_col = outcome + '_T', event_col = outcome + '_E')
                    except:
                        weibull_failed_to_conv_penalizers[l1_ratio].append(penalizer)
                        weibull_failed_to_conv_l1_ratios[penalizer].append(l1_ratio)
                        continue
                    weibull_valid_ci, weibull_valid_mae, weibull_valid_pred_df \
                        = get_ci_mae_preds(weibull_model, valid_df, valid_df_patnos, 'Weibull', outcome, float(num_years))
                    weibull_train_ci, weibull_train_mae, weibull_train_pred_df \
                        = get_ci_mae_preds(weibull_model, train_df, train_df_patnos, 'Weibull', outcome, float(num_years))
                    weibull_valid_auroc, weibull_valid_acc, weibull_valid_prec, weibull_valid_rec \
                        = get_auroc_acc_prec_rec(valid_df[[outcome + '_E']], weibull_valid_pred_df, outcome, \
                                                 float(num_years))
                    weibull_train_auroc, weibull_train_acc, weibull_train_prec, weibull_train_rec \
                        = get_auroc_acc_prec_rec(train_df[[outcome + '_E']], weibull_train_pred_df, outcome, \
                                                 float(num_years))
                    weibull_vary_penalizer_train_metrics[l1_ratio]['auroc'].append(weibull_train_auroc)
                    weibull_vary_penalizer_train_metrics[l1_ratio]['acc'].append(weibull_train_acc)
                    weibull_vary_penalizer_train_metrics[l1_ratio]['prec'].append(weibull_train_prec)
                    weibull_vary_penalizer_train_metrics[l1_ratio]['rec'].append(weibull_train_rec)
                    weibull_vary_penalizer_train_metrics[l1_ratio]['ci'].append(weibull_train_ci)
                    weibull_vary_penalizer_train_metrics[l1_ratio]['mae'].append(weibull_train_mae)
                    weibull_vary_l1_ratio_train_metrics[penalizer]['auroc'].append(weibull_train_auroc)
                    weibull_vary_l1_ratio_train_metrics[penalizer]['acc'].append(weibull_train_acc)
                    weibull_vary_l1_ratio_train_metrics[penalizer]['prec'].append(weibull_train_prec)
                    weibull_vary_l1_ratio_train_metrics[penalizer]['rec'].append(weibull_train_rec)
                    weibull_vary_l1_ratio_train_metrics[penalizer]['ci'].append(weibull_train_ci)
                    weibull_vary_l1_ratio_train_metrics[penalizer]['mae'].append(weibull_train_mae)
                    weibull_vary_penalizer_valid_metrics[l1_ratio]['auroc'].append(weibull_valid_auroc)
                    weibull_vary_penalizer_valid_metrics[l1_ratio]['acc'].append(weibull_valid_acc)
                    weibull_vary_penalizer_valid_metrics[l1_ratio]['prec'].append(weibull_valid_prec)
                    weibull_vary_penalizer_valid_metrics[l1_ratio]['rec'].append(weibull_valid_rec)
                    weibull_vary_penalizer_valid_metrics[l1_ratio]['ci'].append(weibull_valid_ci)
                    weibull_vary_penalizer_valid_metrics[l1_ratio]['mae'].append(weibull_valid_mae)
                    weibull_vary_l1_ratio_valid_metrics[penalizer]['auroc'].append(weibull_valid_auroc)
                    weibull_vary_l1_ratio_valid_metrics[penalizer]['acc'].append(weibull_valid_acc)
                    weibull_vary_l1_ratio_valid_metrics[penalizer]['prec'].append(weibull_valid_prec)
                    weibull_vary_l1_ratio_valid_metrics[penalizer]['rec'].append(weibull_valid_rec)
                    weibull_vary_l1_ratio_valid_metrics[penalizer]['ci'].append(weibull_valid_ci)
                    weibull_vary_l1_ratio_valid_metrics[penalizer]['mae'].append(weibull_valid_mae)
                    if weibull_valid_auroc > weibull_valid_best_metrics['auroc'] \
                        or (weibull_valid_auroc == weibull_valid_best_metrics['auroc'] \
                            and weibull_valid_acc > weibull_valid_best_metrics['acc']) \
                        or (weibull_valid_auroc == weibull_valid_best_metrics['auroc'] \
                            and weibull_valid_acc == weibull_valid_best_metrics['acc'] \
                            and weibull_valid_prec > weibull_valid_best_metrics['prec']) \
                        or (weibull_valid_auroc == weibull_valid_best_metrics['auroc'] \
                            and weibull_valid_acc == weibull_valid_best_metrics['acc'] \
                            and weibull_valid_prec == weibull_valid_best_metrics['prec'] \
                            and weibull_valid_rec > weibull_valid_best_metrics['rec']) \
                        or (weibull_valid_auroc == weibull_valid_best_metrics['auroc'] \
                            and weibull_valid_acc == weibull_valid_best_metrics['acc'] \
                            and weibull_valid_prec == weibull_valid_best_metrics['prec'] \
                            and weibull_valid_rec == weibull_valid_best_metrics['rec'] \
                            and weibull_valid_ci > weibull_valid_best_metrics['ci']) \
                        or (weibull_valid_auroc == weibull_valid_best_metrics['auroc'] \
                            and weibull_valid_acc == weibull_valid_best_metrics['acc'] \
                            and weibull_valid_prec == weibull_valid_best_metrics['prec'] \
                            and weibull_valid_rec == weibull_valid_best_metrics['rec'] \
                            and weibull_valid_ci == weibull_valid_best_metrics['ci'] \
                            and weibull_valid_mae < weibull_valid_best_metrics['mae']):
                        weibull_valid_best_metrics['auroc'] = weibull_valid_auroc
                        weibull_valid_best_metrics['acc'] = weibull_valid_acc
                        weibull_valid_best_metrics['prec'] = weibull_valid_prec
                        weibull_valid_best_metrics['rec'] = weibull_valid_rec
                        weibull_valid_best_metrics['ci'] = weibull_valid_ci
                        weibull_valid_best_metrics['mae'] = weibull_valid_mae
                        weibull_train_best_metrics['auroc'] = weibull_train_auroc
                        weibull_train_best_metrics['acc'] = weibull_train_acc
                        weibull_train_best_metrics['prec'] = weibull_train_prec
                        weibull_train_best_metrics['rec'] = weibull_train_rec
                        weibull_train_best_metrics['ci'] = weibull_train_ci
                        weibull_train_best_metrics['mae'] = weibull_train_mae
                        weibull_best_penalizer = penalizer
                        weibull_best_l1_ratio = l1_ratio
                        weibull_test_ci, weibull_test_mae, weibull_test_pred_df \
                            = get_ci_mae_preds(weibull_model, test_df, test_df_patnos, 'Weibull', outcome, float(num_years))
                        weibull_test_auroc, weibull_test_acc, weibull_test_prec, weibull_test_rec \
                            = get_auroc_acc_prec_rec(test_df[[outcome + '_E']], weibull_test_pred_df, outcome, \
                                                     float(num_years))
                        weibull_test_best_metrics['auroc'] = weibull_test_auroc
                        weibull_test_best_metrics['acc'] = weibull_test_acc
                        weibull_test_best_metrics['prec'] = weibull_test_prec
                        weibull_test_best_metrics['rec'] = weibull_test_rec
                        weibull_test_best_metrics['ci'] = weibull_test_ci
                        weibull_test_best_metrics['mae'] = weibull_test_mae
                        weibull_fold_coefs_best = weibull_model.params_.copy()
                        weibull_test_pred_df.to_csv(output_dir + outcome + '_weibull_fold' + str(fold_idx) + '_test_preds.csv', \
                                                    index=False)
                        weibull_train_valid_pred_df = pd.concat([weibull_train_pred_df, weibull_valid_pred_df])
                        weibull_train_valid_pred_df.to_csv(output_dir + outcome + '_weibull_fold' + str(fold_idx) \
                                                           + '_train_valid_preds.csv', index=False)
            assert weibull_valid_best_metrics['auroc'] != 0
            for metric in metric_list:
                weibull_train_metrics[metric].append(weibull_train_best_metrics[metric])
                weibull_valid_metrics[metric].append(weibull_valid_best_metrics[metric])
                weibull_test_metrics[metric].append(weibull_test_best_metrics[metric])
            weibull_penalizers[0] = 1e-3
            failed_penalizers = weibull_failed_to_conv_penalizers[weibull_best_l1_ratio]
            if len(failed_penalizers) > 0:
                if failed_penalizers[0] == 0:
                    failed_penalizers[0] = 1e-3
                for metric_idx in range(len(metric_list)):
                    weibull_penalizer_ax[metric_idx, fold_idx].plot(failed_penalizers, np.zeros(len(failed_penalizers)), c='r', \
                                                                    marker='x')
                for penalizer in failed_penalizers:
                    weibull_penalizers.remove(penalizer)
            if weibull_best_penalizer == 0:
                best_penalizer_scatter_pt = 1e-3
            else:
                best_penalizer_scatter_pt = weibull_best_penalizer
            for metric_idx in range(len(metric_list)):
                metric = metric_list[metric_idx]
                curr_ax = weibull_penalizer_ax[metric_idx, fold_idx]
                curr_ax.plot(weibull_penalizers, weibull_vary_penalizer_train_metrics[weibull_best_l1_ratio][metric], \
                             'b', linestyle='--', label='train')
                curr_ax.plot(weibull_penalizers, weibull_vary_penalizer_valid_metrics[weibull_best_l1_ratio][metric], \
                             'r', label='valid')
                curr_ax.scatter([best_penalizer_scatter_pt], [weibull_train_best_metrics[metric]], c='b')
                curr_ax.scatter([best_penalizer_scatter_pt], [weibull_valid_best_metrics[metric]], c='r')
                curr_ax.set_xscale('log')
                curr_ax.set_xlabel('Penalizer')
                curr_ax.set_ylabel(metric_human_readable_list[metric_idx])
                curr_ax.set_title('Fold ' + str(fold_idx))
            failed_l1_ratios = weibull_failed_to_conv_l1_ratios[weibull_best_penalizer]
            if len(failed_l1_ratios) > 0:
                for metric_idx in range(len(metric_list)):
                    weibull_l1_ratio_ax[metric_idx, fold_idx].plot(failed_l1_ratios, np.zeros(len(failed_l1_ratios)), c='r', \
                                                                   marker='x')
                for l1_ratio in failed_l1_ratios:
                    weibull_l1_ratios.remove(l1_ratio)
            for metric_idx in range(len(metric_list)):
                metric = metric_list[metric_idx]
                curr_ax = weibull_l1_ratio_ax[metric_idx, fold_idx]
                curr_ax.plot(weibull_l1_ratios, weibull_vary_l1_ratio_train_metrics[weibull_best_penalizer][metric], \
                             'b', linestyle='--', label='train')
                curr_ax.plot(weibull_l1_ratios, weibull_vary_l1_ratio_valid_metrics[weibull_best_penalizer][metric], \
                             'r', label='valid')
                curr_ax.scatter([weibull_best_l1_ratio], [weibull_train_best_metrics[metric]], c='b')
                curr_ax.scatter([weibull_best_l1_ratio], [weibull_valid_best_metrics[metric]], c='r')
                curr_ax.set_xlabel('L1 ratio')
                curr_ax.set_ylabel(metric_human_readable_list[metric_idx])
                curr_ax.set_title('Fold ' + str(fold_idx))
            weibull_fold_coefs_best = weibull_fold_coefs_best.reset_index()
            weibull_fold_coefs_best = weibull_fold_coefs_best.loc[weibull_fold_coefs_best['level_1'] != '_intercept']
            del weibull_fold_coefs_best['level_0']
            weibull_fold_coefs_best.rename(columns={'level_1': 'Feature', 0:  'coef_fold' + str(fold_idx)}, \
                                           inplace=True)
            weibull_coefs = weibull_coefs.merge(weibull_fold_coefs_best, on=['Feature'], validate='one_to_one', \
                                                suffixes=(False, False))
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
        cox_outcome_metrics[outcome] = cox_test_metrics
        weibull_outcome_metrics[outcome] = weibull_test_metrics

    with open(output_dir + outcome + 'cox_test_metrics.pkl', 'wb') as f:
        pickle.dump(cox_outcome_metrics, f, protocol=2)
    with open(output_dir + outcome + 'weibull_test_metrics.pkl', 'wb') as f:
        pickle.dump(weibull_outcome_metrics, f, protocol=2)
    metrics_str = 'Cox:\n'
    for outcome in outcomes:
        metrics_str += outcome + '\n'
        for metric in metric_list:
            metrics_str += metric + ': {0:.4f}'.format(np.mean(np.array(cox_outcome_metrics[outcome][metric]))) \
                + ' ({0:.4f})\n'.format(np.std(np.array(cox_outcome_metrics[outcome][metric]))) 
    metrics_str += 'Weibull:\n'
    for outcome in outcomes:
        metrics_str += outcome + '\n'
        for metric in metric_list:
            metrics_str += metric + ': {0:.4f}'.format(np.mean(np.array(weibull_outcome_metrics[outcome][metric]))) \
                + ' ({0:.4f})\n'.format(np.std(np.array(weibull_outcome_metrics[outcome][metric])))
    with open(output_dir + 'survival_test_metrics.txt', 'w') as f:
        f.write(metrics_str)

    with open(output_dir + 'sample_sizes.pkl', 'wb') as f:
        pickle.dump(outcome_num_patnos, f, protocol=2)
    sample_sizes_str = ''
    for outcome in outcome_num_patnos.keys():
        sample_sizes_str += outcome + ': ' + str(outcome_num_patnos[outcome]) + '\n'
    with open(output_dir + 'sample_sizes.txt', 'w') as f:
        f.write(sample_sizes_str)
        
if __name__ == '__main__':
    main()