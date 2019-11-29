import numpy as np, pandas as pd, pickle, sys, os, matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression

param_msg = 'Expecting 1st parameter to be name of covariate set and 2nd parameter to be length of trial period in years.'
if len(sys.argv) != 3:
    print(param_msg)
    sys.exit()
covariate_set_name = sys.argv[1]
num_years = sys.argv[2]
if num_years not in {'2', '3'}:
    print(param_err_msg)
    print('Allowable options for 2nd parameter: 2 or 3')
    sys.exit()
with open('finalized_covariate_sets_filter' + num_years + '.pkl', 'rb') as f:
    baseline_feat_dicts = pickle.load(f, encoding='latin1')
if covariate_set_name not in baseline_feat_dicts.keys():
    print(param_err_msg)
    print('Allowable options for 1st parameter:')
    print(baseline_feat_dicts.keys())
    sys.exit()
covariate_sets = baseline_feat_dicts[covariate_set_name]
surv_df_filtered = pd.read_csv('PD_outcomes_filter' + num_years + 'yrs.csv')
with open('test_patnos_filter' + num_years + '.pkl', 'rb') as f:
    filtered_test_patnos = pickle.load(f, encoding='latin1')
#with open('feats_to_remove_filter' + num_years + '.pkl', 'rb') as f:
#    feats_to_remove = pickle.load(f, encoding='latin1')
with open('valid_patnos_filter' + num_years + '.pkl', 'rb') as f:
    valid_patnos_dict = pickle.load(f, encoding='latin1')
baseline_df = pd.read_csv('../finalized_outcome_survival_models/final_survival_baseline_data.csv')
output_dir = 'logistic_regression_' + num_years + '_years_' + covariate_set_name + '_2019Jul30_testset_fixed/'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
with open('../finalized_outcome_survival_models/final_human_readable_feat_dict.pkl', 'rb') as f:
    human_readable_feat_dict = pickle.load(f, encoding='latin1')

outcomes = ['Autonomic', 'Cognitive', 'Psychiatric', 'Motor', 'Sleep', 'hybrid_requiremotor', 'NUPDRS23_45', 'MOCA_25', \
            'MSEADLG_79']
metric_list = ['auroc', 'acc', 'prec', 'rec']
metric_human_readable_list = ['AUROC', 'Accuracy', 'Precision', 'Recall']
output_str = ''
for outcome in outcomes:
    output_str += outcome + '\n'
    '''
    if outcome == 'NUPDRS23_45':
        outcome_cov_set = list(covariate_sets['Standard']) + list(covariate_sets['Motor'])
    elif outcome == 'MOCA_25':
        outcome_cov_set = list(covariate_sets['Standard']) + list(covariate_sets['Cognitive'])
    elif outcome == 'MSEADLG_79' or outcome == 'hybrid_requiremotor':
        outcome_cov_set = list(covariate_sets['Standard']) + list(covariate_sets['Autonomic']) \
            + list(covariate_sets['Cognitive']) + list(covariate_sets['Psychiatric']) + list(covariate_sets['Motor']) \
            + list(covariate_sets['Sleep'])
    else:
        outcome_cov_set = list(covariate_sets[outcome])
    outcome_cov_set = list(set(outcome_cov_set).difference(set(feats_to_remove[outcome])))
    '''
    outcome_cov_set = list(covariate_sets[outcome])
    fold_test_metrics_dict = {'auroc': [], 'acc': [], 'prec': [], 'rec': []}
    fold_coefs = pd.DataFrame({'Feature': outcome_cov_set})
    fold_vary_l1_ratio_fig, fold_vary_l1_ratio_ax = plt.subplots(nrows=4, ncols=4, figsize=(20,20), sharex=True, \
                                                                 sharey=True)
    fold_vary_C_fig, fold_vary_C_ax = plt.subplots(nrows=4, ncols=4, figsize=(20,20), sharex=True, sharey=True)
    outcome_test_patnos = filtered_test_patnos[outcome]
    outcome_surv_df_filtered = surv_df_filtered[['PATNO', outcome + '_T', outcome + '_E']]
    outcome_df = baseline_df[['PATNO'] + outcome_cov_set]
    outcome_df = outcome_surv_df_filtered.merge(outcome_df, on=['PATNO'], validate='one_to_one').dropna()
    assert set(outcome_test_patnos.tolist()).issubset(set(outcome_df.PATNO.values.tolist()))
    test_outcome_df = outcome_df.loc[outcome_df['PATNO'].isin(filtered_test_patnos[outcome])]
    train_valid_outcome_df = outcome_df.loc[~outcome_df['PATNO'].isin(filtered_test_patnos[outcome])]
    test_X = test_outcome_df[outcome_cov_set].values
    test_Y = test_outcome_df[outcome + '_E'].values
    test_patnos_arr = test_outcome_df['PATNO'].values
    train_valid_patnos = train_valid_outcome_df.PATNO.values
    for fold_idx in range(4):
        print(outcome + ' ' + str(fold_idx))
        valid_patnos = valid_patnos_dict[outcome][fold_idx]
        train_patnos = set(train_valid_patnos.tolist()).difference(set(valid_patnos.tolist()))
        train_outcome_df = train_valid_outcome_df.loc[train_valid_outcome_df['PATNO'].isin(train_patnos)]
        valid_outcome_df = train_valid_outcome_df.loc[train_valid_outcome_df['PATNO'].isin(valid_patnos)]
        train_X = train_outcome_df[outcome_cov_set].values
        train_Y = train_outcome_df[outcome + '_E'].values
        valid_X = valid_outcome_df[outcome_cov_set].values
        valid_Y = valid_outcome_df[outcome + '_E'].values
        l1_ratios = [0., 0.5, 1.0]
        Cs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1., 5., 10., 50., 100., 500., 1000., 5000., 1e4]
        best_valid_auroc = 0.
        best_valid_acc = 0.
        best_valid_prec = 0.
        best_valid_rec = 0.
        train_vary_l1_ratio_metrics_dict = dict()
        train_vary_C_metrics_dict = dict()
        valid_vary_l1_ratio_metrics_dict = dict()
        valid_vary_C_metrics_dict = dict()
        for l1_ratio in l1_ratios:
            train_vary_C_metrics_dict[l1_ratio] = {'auroc': [], 'acc': [], 'prec': [], 'rec': []}
            valid_vary_C_metrics_dict[l1_ratio] = {'auroc': [], 'acc': [], 'prec': [], 'rec': []}
        for C in Cs:
            train_vary_l1_ratio_metrics_dict[C] = {'auroc': [], 'acc': [], 'prec': [], 'rec': []}
            valid_vary_l1_ratio_metrics_dict[C] = {'auroc': [], 'acc': [], 'prec': [], 'rec': []}
            for l1_ratio in l1_ratios:
                log_reg = LogisticRegression(penalty='elasticnet', l1_ratio=l1_ratio, C=C, max_iter=1e5, solver='saga', \
                                             random_state=0)
                log_reg.fit(train_X, train_Y)
                train_pred = log_reg.predict(train_X)
                train_prob = log_reg.predict_proba(train_X)[:,1]
                valid_pred = log_reg.predict(valid_X)
                valid_prob = log_reg.predict_proba(valid_X)[:,1]
                valid_auroc = roc_auc_score(valid_Y, valid_prob)
                valid_acc = accuracy_score(valid_Y, valid_pred)
                valid_prec = precision_score(valid_Y, valid_pred)
                valid_rec = recall_score(valid_Y, valid_pred)
                train_auroc = roc_auc_score(train_Y, train_prob)
                train_acc = accuracy_score(train_Y, train_pred)
                train_prec = precision_score(train_Y, train_pred)
                train_rec = recall_score(train_Y, train_pred)
                train_vary_l1_ratio_metrics_dict[C]['auroc'].append(train_auroc)
                train_vary_l1_ratio_metrics_dict[C]['acc'].append(train_acc)
                train_vary_l1_ratio_metrics_dict[C]['prec'].append(train_prec)
                train_vary_l1_ratio_metrics_dict[C]['rec'].append(train_rec)
                train_vary_C_metrics_dict[l1_ratio]['auroc'].append(train_auroc)
                train_vary_C_metrics_dict[l1_ratio]['acc'].append(train_acc)
                train_vary_C_metrics_dict[l1_ratio]['prec'].append(train_prec)
                train_vary_C_metrics_dict[l1_ratio]['rec'].append(train_rec)
                valid_vary_l1_ratio_metrics_dict[C]['auroc'].append(valid_auroc)
                valid_vary_l1_ratio_metrics_dict[C]['acc'].append(valid_acc)
                valid_vary_l1_ratio_metrics_dict[C]['prec'].append(valid_prec)
                valid_vary_l1_ratio_metrics_dict[C]['rec'].append(valid_rec)
                valid_vary_C_metrics_dict[l1_ratio]['auroc'].append(valid_auroc)
                valid_vary_C_metrics_dict[l1_ratio]['acc'].append(valid_acc)
                valid_vary_C_metrics_dict[l1_ratio]['prec'].append(valid_prec)
                valid_vary_C_metrics_dict[l1_ratio]['rec'].append(valid_rec)
                if valid_auroc > best_valid_auroc or (valid_auroc == best_valid_auroc and valid_acc > best_valid_acc) \
                    or (valid_auroc == best_valid_auroc and valid_acc == best_valid_acc and valid_prec > best_valid_prec) \
                    or (valid_auroc == best_valid_auroc and valid_acc == best_valid_acc and valid_prec == best_valid_prec \
                        and valid_rec > best_valid_rec):
                    best_valid_auroc = valid_auroc
                    best_valid_acc = valid_acc
                    best_valid_prec = valid_prec
                    best_valid_rec = valid_rec
                    best_train_metrics_list = [train_auroc, train_acc, train_prec, train_rec]
                    best_l1_ratio = l1_ratio
                    best_C = C
                    best_test_pred = log_reg.predict(test_X)
                    best_test_prob = log_reg.predict_proba(test_X)[:,1]
                    best_coefs = log_reg.coef_.flatten()
                    best_test_pred_df = pd.DataFrame({'PATNO': test_patnos_arr, 'pred': best_test_pred, 'prob': best_test_prob})
                    best_test_pred_df.to_csv(output_dir + outcome + '_test_preds_fold' + str(fold_idx) + '.csv', index=False)
        best_valid_metrics_list = [best_valid_auroc, best_valid_acc, best_valid_prec, best_valid_rec]
        for metric_idx in range(len(metric_list)):
            metric = metric_list[metric_idx]
            fold_vary_l1_ratio_ax[metric_idx, fold_idx].plot(l1_ratios, train_vary_l1_ratio_metrics_dict[best_C][metric], \
                                                             c='b', linestyle='--', label='train')
            fold_vary_l1_ratio_ax[metric_idx, fold_idx].scatter(best_l1_ratio, best_train_metrics_list[metric_idx], \
                                                                c='b')
            fold_vary_l1_ratio_ax[metric_idx, fold_idx].plot(l1_ratios, valid_vary_l1_ratio_metrics_dict[best_C][metric], \
                                                             c='r', label='valid')
            fold_vary_l1_ratio_ax[metric_idx, fold_idx].scatter(best_l1_ratio, best_valid_metrics_list[metric_idx], \
                                                                c='r')
            fold_vary_l1_ratio_ax[metric_idx, fold_idx].set_xlabel('L1 ratio')
            fold_vary_l1_ratio_ax[metric_idx, fold_idx].set_ylabel(metric_human_readable_list[metric_idx])
            fold_vary_l1_ratio_ax[metric_idx, fold_idx].set_title('Fold ' + str(fold_idx))
            fold_vary_C_ax[metric_idx, fold_idx].plot(Cs, train_vary_C_metrics_dict[best_l1_ratio][metric], c='b', \
                                                      linestyle='--', label='train')
            fold_vary_C_ax[metric_idx, fold_idx].scatter(best_C, best_train_metrics_list[metric_idx], c='b')
            fold_vary_C_ax[metric_idx, fold_idx].plot(Cs, valid_vary_C_metrics_dict[best_l1_ratio][metric], c='r', \
                                                      label='valid')
            fold_vary_C_ax[metric_idx, fold_idx].scatter(best_C, best_valid_metrics_list[metric_idx], c='r')
            fold_vary_C_ax[metric_idx, fold_idx].set_xlabel('C')
            fold_vary_C_ax[metric_idx, fold_idx].set_ylabel(metric_human_readable_list[metric_idx])
            fold_vary_C_ax[metric_idx, fold_idx].set_title('Fold ' + str(fold_idx))
            fold_vary_C_ax[metric_idx, fold_idx].set_xscale('log')
        fold_coefs['coef_fold' + str(fold_idx)] = best_coefs
        fold_test_metrics_dict['auroc'].append(roc_auc_score(test_Y, best_test_prob))
        fold_test_metrics_dict['acc'].append(accuracy_score(test_Y, best_test_pred))
        fold_test_metrics_dict['prec'].append(precision_score(test_Y, best_test_pred))
        fold_test_metrics_dict['rec'].append(recall_score(test_Y, best_test_pred))
    for metric_idx in range(len(metric_list)):
        fold_vary_l1_ratio_ax[metric_idx, 3].legend()
        fold_vary_C_ax[metric_idx, 3].legend()
    fold_vary_l1_ratio_fig.tight_layout()
    fold_vary_l1_ratio_fig.savefig(output_dir + outcome + '_vary_l1_ratio.pdf')
    fold_vary_C_fig.tight_layout()
    fold_vary_C_fig.savefig(output_dir + outcome + '_vary_C.pdf')
    
    fold_coefs['coef_mean'] = fold_coefs[['coef_fold' + str(fold_idx) for fold_idx in range(4)]].mean(axis=1)
    fold_coefs['coef_std'] = fold_coefs[['coef_fold' + str(fold_idx) for fold_idx in range(4)]].std(axis=1)
    fold_coefs = fold_coefs.sort_values(by='coef_mean')
    fold_coefs.to_csv(output_dir + outcome + '_coefs.csv', index=False)
    coef_fig, coef_ax = plt.subplots(figsize=(9, len(fold_coefs)*.5+2))
    coef_ax.errorbar(fold_coefs['coef_mean'].values, range(len(fold_coefs)), xerr=fold_coefs['coef_std'].values, fmt='o', \
                     capsize=3)
    coef_ax.axvline(x=0, color='black')
    def reformat_coef_labels(labels):
        formatted_labels = []
        for label in labels:
            formatted_labels.append(human_readable_feat_dict[label])
        return formatted_labels
    coef_ax.set_yticks(ticks=range(len(fold_coefs)))
    coef_ax.set_yticklabels(reformat_coef_labels(fold_coefs['Feature'].values))
    coef_ax.set_xlabel('Logistic regression coefficients')
    coef_fig.tight_layout()
    coef_fig.savefig(output_dir + outcome + '_coefs.pdf')
        
    output_str += 'AUROC: {0:.4f}'.format(np.mean(np.array(fold_test_metrics_dict['auroc']))) \
        + ' ({0:.4f})\n'.format(np.std(np.array(fold_test_metrics_dict['auroc'])))
    output_str += 'Accuracy: {0:.4f}'.format(np.mean(np.array(fold_test_metrics_dict['acc']))) \
        + ' ({0:.4f})\n'.format(np.std(np.array(fold_test_metrics_dict['acc'])))
    output_str += 'Precision: {0:.4f}'.format(np.mean(np.array(fold_test_metrics_dict['prec']))) \
        + ' ({0:.4f})\n'.format(np.std(np.array(fold_test_metrics_dict['prec'])))
    output_str += 'Recall: {0:.4f}'.format(np.mean(np.array(fold_test_metrics_dict['rec']))) \
        + ' ({0:.4f})\n'.format(np.std(np.array(fold_test_metrics_dict['rec'])))
with open(output_dir + 'test_metrics.txt', 'w') as f:
    f.write(output_str)