import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from scipy.stats import pearsonr 
from sklearn import metrics

import sys, os
DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DIR_PATH)

from DataLoaders.PPMI_loader import PPMI_loader
from Synthesize_data import SYN_DATA
from Evaluation.RankingEvaluator import RankingEvaluator
from PostLatentModels.CorrelationCalculator import CorrelationCalculator
from Evaluation.MeanSquaredErrorEvaluator import MeanSquaredErrorEvaluator
from PostLatentModels.LatentRateCalculator import LatentRateCalculator
from PostLatentModels.LatentRateClusterClassifier import LatentRateClusterClassifier

def process_df_for_aging_model(df):
    """
    assume PATNO, EVENT_BY_DUR, and DIS_DUR_BY_CONSENTDT are columns in df
    drop EVENT_ID_DUR
    rename PATNO -> individual_id and DIS_DUR_BY_CONSENTDT -> age_sex__age
    drop row with DIS_DUR_BY_CONSENTDT = 0
    """
    df = df.rename(columns={"PATNO":"individual_id", "DIS_DUR_BY_CONSENTDT":"age_sex___age"}).copy()
    df = df.drop(["EVENT_ID_DUR"], axis=1)
    df.loc[:,"individual_id"] = df.index
    df = df.loc[df["age_sex___age"] > 0]
    return df

def plot_dist_age_latent(model, df, num_r, num_bin=100):
    """
    plot distrubution of the latent factors learned from model
    model - Pierson's Aging model
    df - dataframe in Aging format
    num_r - number of latent rate from model
    num_bin - number of bins to plot the distribution of the latents
    """

    estimated_Zs = model.get_projections(df, project_onto_mean=True)
    estimated_Zs = estimated_Zs.drop('individual_id', 1)
    estimated_rb = estimated_Zs.values
    for i in range(num_r):
        preprocessed_ages = model.age_preprocessing_function(df['age_sex___age'])
        estimated_r = estimated_Zs['z%i' % i] / preprocessed_ages
        plt.hist(estimated_r, bins=num_bin)
        plt.xlabel('Estimated aging rate ' + str(i))
        plt.show()
        
        feature_pearson = list()
        for feature in list(df.columns):
            if feature not in ["individual_id", "age_sex___age", "EVENT_ID_DUR"]:
                pear_corr, _ = pearsonr(df[feature].values, estimated_r)
                feature_pearson.append((feature, pear_corr))
                
        feature_pearson.sort(key=lambda x: x[1], reverse=True)
        for feature, pear_corr in feature_pearson[:5]:
            print(feature, pear_corr)


def get_merge_df(aging_model, aging_df, normal_df, know_age=False):
    aging_df = aging_df.copy()
    aging_df_age_1 = aging_df.copy()
    if not know_age:
        aging_df_age_1.loc[:, "age_sex___age"] = 1.0
    estimated_Zs= aging_model.get_projections(aging_df_age_1, project_onto_mean=True)
    merge_df = pd.merge(normal_df, estimated_Zs, left_index=True, right_index=True)

    return merge_df

def get_CI(aging_model, aging_df, normal_df, latent_col):
    """
    get the concordance index of the latent
    factors learned by aging_model

    aging_model - model in Aging format
    aging_df - dataframe in Aging format
    normal_df - dataframe in normal format (from PPMI data_loader)
    latent_col - name of the latent factor of interest
    """
    merge_df = get_merge_df(aging_model, aging_df, normal_df)
    CI = RankingEvaluator().get_concordance_index(merge_df, [latent_col])
    return CI

def get_CI_and_consec_CI(aging_model, aging_df, normal_df, latent_col):
    """
    get the concordance index and the consecutive concordance index of the latent
    factors learned by aging_model

    aging_model - model in Aging format
    aging_df - dataframe in Aging format
    normal_df - dataframe in normal format (from PPMI data_loader)
    latent_col - name of the latent factor of interest
    """
    merge_df = get_merge_df(aging_model, aging_df, normal_df)
    CI = RankingEvaluator().get_concordance_index(merge_df, [latent_col])
    consec_CI = RankingEvaluator().get_consec_visit_ranking(merge_df, latent_col)
    return (CI, consec_CI)

def best_model(model_list, aging_df, normal_df, latent_col):
    """
    choose the model that achieves highest concordance index and calculate
    concordance index, consecutive concordance index and the mean square error
    
    model_list - list of models that are already trained
    aging_df - dataframe in Aging format
    normal_df - dataframe in normal format (from PPMI data_loader)
    latent_col - name of the latent factor of interest
    """
    all_CI = list()
    all_consec_CI = list()
    all_mse = list()
    for model in model_list:
        CI, consec_CI = get_CI_and_consec_CI(model, aging_df, normal_df, latent_col)
        mse = get_mse_result(model, aging_df, latent_col)
        all_CI.append(CI)
        all_consec_CI.append(consec_CI)
        all_mse.append(mse)

    best_model_index = np.argmax(all_CI)

    return_dict=dict()
    return_dict["all_CI"] = all_CI
    return_dict["all_consec_CI"] = all_consec_CI
    return_dict["all_mse"] = all_mse   
    return_dict["best_model"] = model_list[best_model_index] 

    return return_dict


def get_mse_result(aging_model, aging_df, subsets=None):
    """
    get mean square error of the reconstructed input vector by aging_model

    aging_model - model in Aging format
    aging_df - dataframe in Aging format
    subsets - dictionary mapping subset name to subset columns
    """
    estimated_Zs = aging_model.get_projections(aging_df, project_onto_mean=True)
    pred = aging_model.reconstruct_data(estimated_Zs).drop(columns=["individual_id"])
    true = aging_df.drop(columns=["individual_id", "age_sex___age"])
    mse = np.mean(np.square(true.values - pred.values))
    
    if subsets is not None:
        subset_mses = dict()
        for subset_name in subsets.keys():
            assert set(subsets[subset_name]).issubset(set(true.columns.values.tolist()))
            assert set(subsets[subset_name]).issubset(set(pred.columns.values.tolist()))
            subset_mses[subset_name] \
                = np.mean(np.square(true[list(subsets[subset_name])].values - pred[list(subsets[subset_name])].values))
        return mse, subset_mses
    return mse

def get_mse2(aging_model, aging_df, normal_df, age_col, latent_column_names, observed_column_names, fit_intercept=True):
    """
    calculate mean square error of the predicted raw features of the future time point

    aging_model - model in Aging format
    aging_df - dataframe in Aging format
    normal_df - dataframe in normal format (from PPMI data_loader)
    age_col - column name of the age variable
    latent_column_names - column names of the latent variables
    observed_column_names - column names of the raw features
    fit_intercept - whether to fit linear regression with bias to the latent variables
    """

    rate_calculator = LatentRateCalculator()
    mse_evaluator = MeanSquaredErrorEvaluator()

    test_df = get_merge_df(aging_model, aging_df, normal_df, know_age=True)
    old_test_df = test_df

    test_df = test_df.copy().drop(columns=["EVENT_ID_DUR"])
    test_df = test_df.rename(columns={"DIS_DUR_BY_CONSENTDT": "EVENT_ID_DUR"})
    age_col = "EVENT_ID_DUR"

    test_patno_counts = dict(test_df.PATNO.value_counts())
    test_patnos_atleast_3timepoints = set()
    for patno in test_patno_counts.keys():
        if test_patno_counts[patno] >= 3:
            test_patnos_atleast_3timepoints.add(patno)
    test_data_atleast_3timepoints = test_df.loc[test_df['PATNO'].isin(test_patnos_atleast_3timepoints)]
    test_data_first_half_patnos = []
    test_data_first_half_event_id_durs = []
    test_data_second_half_patnos = []
    test_data_second_half_event_id_durs = []
    for patno in test_patnos_atleast_3timepoints:
        patno_df = test_df.loc[test_df['PATNO']==patno]
        patno_df = patno_df.sort_values(by=[age_col])
        first_half_end_idx = int(math.ceil(len(patno_df)/2.))
        test_data_first_half_patnos += first_half_end_idx*[patno]
        test_data_first_half_event_id_durs += patno_df.EVENT_ID_DUR.values[:first_half_end_idx].tolist()
        test_data_second_half_patnos += (len(patno_df)-first_half_end_idx)*[patno]
        test_data_second_half_event_id_durs += patno_df.EVENT_ID_DUR.values[first_half_end_idx:].tolist()
    test_data_first_half = pd.DataFrame({'PATNO': test_data_first_half_patnos, age_col: test_data_first_half_event_id_durs})
    test_df = test_df.drop_duplicates(subset=['PATNO', age_col])
    test_data_first_half = test_data_first_half.drop_duplicates(subset=['PATNO', age_col])
    test_data_first_half_latents = test_data_first_half.merge(test_df[['PATNO', age_col]+latent_column_names], \
                                                              on=['PATNO', age_col], validate='one_to_one')
    test_data_first_half_rates_biases = rate_calculator.get_rate_of_latent(test_data_first_half_latents, 
                                                                           latent_column_names, fit_intercept=fit_intercept)
    test_data_second_half = pd.DataFrame({'PATNO': test_data_second_half_patnos, \
                                          age_col: test_data_second_half_event_id_durs})
    test_data_second_half = test_data_second_half.drop_duplicates(subset=['PATNO', age_col])
    test_data_second_half_observeds = test_data_second_half.merge(test_df[['PATNO', age_col]+observed_column_names], \
                                                                  on=['PATNO', age_col], validate='one_to_one')

    test_data_second_half_observeds.drop(columns=[age_col, "PATNO"], inplace=True)
    # extrapolate, decode, and get MSE
    test_data_second_half_latents = rate_calculator.extrapolate_latent_factors(test_data_first_half_rates_biases, \
                                                                               test_data_second_half, latent_column_names)

    test_data_second_half_latents.drop(columns=[age_col, "PATNO"], inplace=True)
    test_data_second_half_latents.insert(0, "individual_id", test_data_second_half_latents.index)
    test_data_second_half_preds = aging_model.reconstruct_data(test_data_second_half_latents).drop(columns=["individual_id"])


    mse = np.mean(np.square(test_data_second_half_observeds.values - test_data_second_half_preds.values))

    return mse


def subtyping_aging(data_dict, model, latent_cols, sampled=False, fit_intercept=True, pred_dir=None):
    """
    Calculating AUROC, acc, precisions and recalls of subtyping task.

    data_dict - a dictionary of data that is generated from data_loader
    model - a trained aging model
    latent-cols - column names of the latent variables
    sampled - whether data_dict is generated with sampled data
    fit_intercept - whether to fit linear regression with bias to the latent variables
    pred_dir - directory to save the result
    """


    data_loader = PPMI_loader("PD_extended_questions.csv")

    train_valid_test_patnos = dict()

    if sampled:
        train_valid_test_patnos['train'] = set(data_dict["orig_train_df"].PATNO.unique().tolist())
    else:
        train_valid_test_patnos['train'] = set(data_dict["train_df"].PATNO.unique().tolist())

    train_valid_test_patnos['valid'] = set(data_dict["val_df"].PATNO.unique().tolist())
    train_valid_test_patnos['test'] = set(data_dict["test_df"].PATNO.unique().tolist())

    train_baseline, valid_baseline, test_baseline, baseline_column_names \
            = data_loader.get_baseline_data_split("../../PD_selected_baseline.csv", train_valid_test_patnos)
    baseline_human_readable_dict = data_loader.get_baseline_human_readable_dict()

    if sampled:
        merge_train_df = get_merge_df(model, data_dict["aging_orig_train_df"], data_dict["orig_train_df"], know_age=True)
    else:
        merge_train_df = get_merge_df(model, data_dict["aging_train_df"], data_dict["train_df"], know_age=True)

    merge_valid_df = get_merge_df(model, data_dict["aging_val_df"], data_dict["val_df"])
    merge_test_df = get_merge_df(model, data_dict["aging_test_df"], data_dict["test_df"])

    rate_calculator = LatentRateCalculator()

    merge_train_df = rate_calculator.get_atleast_2timepoints_patnos_dataframe(merge_train_df)
    merge_valid_df = rate_calculator.get_atleast_2timepoints_patnos_dataframe(merge_valid_df)
    merge_test_df = rate_calculator.get_atleast_2timepoints_patnos_dataframe(merge_test_df)

    merge_train_df = merge_train_df.drop(columns=["EVENT_ID_DUR"]).rename(columns={"DIS_DUR_BY_CONSENTDT":"EVENT_ID_DUR"})

    train_latent_rates_biases = rate_calculator.get_rate_of_latent(merge_train_df, 
                                                                   latent_cols,
                                                                   fit_intercept=fit_intercept)

    valid_latent_rates_biases = rate_calculator.get_rate_of_latent(merge_valid_df, 
                                                                   latent_cols,
                                                                   fit_intercept=fit_intercept)

    test_latent_rates_biases = rate_calculator.get_rate_of_latent(merge_test_df, 
                                                                  latent_cols,
                                                                  fit_intercept=fit_intercept)

    rate_cluster_classifier = LatentRateClusterClassifier()
    latent_rate_cols = [item + "_rate" for item in latent_cols]
    train_valid_latent_rates_biases = train_latent_rates_biases.append(valid_latent_rates_biases)
    rate_cluster_classifier.find_quantiles_rate_of_latent(train_valid_latent_rates_biases,'z0_rate',2)
    train_latent_rates_biases_clusters \
        = rate_cluster_classifier.split_by_preset_divisions(train_latent_rates_biases, 'z0_rate')
    valid_latent_rates_biases_clusters \
        = rate_cluster_classifier.split_by_preset_divisions(valid_latent_rates_biases, 'z0_rate')
    test_latent_rates_biases_clusters \
        = rate_cluster_classifier.split_by_preset_divisions(test_latent_rates_biases, 'z0_rate')

    # merge clusters with baseline features
    train_clusters_baseline = train_latent_rates_biases_clusters.merge(train_baseline, on=['PATNO'], validate='one_to_one')
    valid_clusters_baseline = valid_latent_rates_biases_clusters.merge(valid_baseline, on=['PATNO'], validate='one_to_one')
    test_clusters_baseline = test_latent_rates_biases_clusters.merge(test_baseline, on=['PATNO'], validate='one_to_one')

    # normalize baseline features using min-max in train (instead of z so binary unaffected)
    baseline_feat_mins = train_clusters_baseline[baseline_column_names].min(axis=0)
    baseline_feat_maxs = train_clusters_baseline[baseline_column_names].max(axis=0)
    assert np.sum(np.where(baseline_feat_maxs - baseline_feat_mins == 0, 1, 0)) == 0
    for idx in range(len(baseline_column_names)):
        col = baseline_column_names[idx]
        train_clusters_baseline[col] = (train_clusters_baseline[col] - baseline_feat_mins[idx]) \
                                       /float(baseline_feat_maxs[col] - baseline_feat_mins[idx])
        valid_clusters_baseline[col] = (valid_clusters_baseline[col] - baseline_feat_mins[idx]) \
                                       /float(baseline_feat_maxs[col] - baseline_feat_mins[idx])
        test_clusters_baseline[col] = (test_clusters_baseline[col] - baseline_feat_mins[idx]) \
                                      /float(baseline_feat_maxs[col] - baseline_feat_mins[idx])

    # fit a classifier + predict using it
    rate_cluster_classifier.fit_rate_clusters_classifier(train_clusters_baseline, valid_clusters_baseline, 'cluster', \
                                                         baseline_column_names)
    X = test_clusters_baseline[baseline_column_names].values
    y_true = test_clusters_baseline['cluster'].values
    y_proba = rate_cluster_classifier.classifier.predict_proba(X)
    y_pred = rate_cluster_classifier.classifier.predict(X)
    auroc = metrics.roc_auc_score(y_true, y_proba[:,1])
    acc = metrics.accuracy_score(y_true, y_pred)
    per_class_precisions = metrics.precision_score(y_true, y_pred, average=None)
    per_class_recalls = metrics.recall_score(y_true, y_pred, average=None)

    if pred_dir is not None:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        file_path = os.path.join(pred_dir, "classifier_coeffs.txt")
        rate_cluster_classifier.print_top_coeffs(file_path, baseline_human_readable_dict)
        with open(file_path, "a") as f:
            print("", file=f)
            print("Accuracy={}".format(acc), file=f)
            print("AUROC={}".format(auroc), file=f)
            print("Precisions={}".format(per_class_precisions), file=f)
            print("Recall={}".format(per_class_recalls), file=f)

    return_dict = dict()
    return_dict["accuracy"] = acc
    return_dict["auroc"] = auroc
    return_dict["precisions"] = per_class_precisions
    return_dict["recalls"] = per_class_recalls

    return return_dict

def get_obs_to_latent_corr(aging_model, aging_df, normal_df, latent_column_names, observed_column_names):
    """
    calculate top 5 observed features associated with each latent factor (pos + neg)

    aging_model - model in Aging format
    aging_df - dataframe in Aging format
    normal_df - dataframe in normal format (from PPMI data_loader)
    latent_column_names - column names of the latent variables
    observed_column_names - column names of the raw features
    """
    merged_df = get_merge_df(aging_model, aging_df, normal_df)
    corr_matrix = np.empty((len(latent_column_names), len(observed_column_names)))
    output_str = ''
    for latent_idx in range(len(latent_column_names)):
        for obs_idx in range(len(observed_column_names)):
            corr_matrix[latent_idx, obs_idx], _ \
                = pearsonr(merged_df[latent_column_names[latent_idx]].values, merged_df[observed_column_names[obs_idx]].values)
        sorted_idxs = np.argsort(np.abs(corr_matrix[latent_idx]))
        output_str += latent_column_names[latent_idx] + ' correlated features: '
        for i in range(10):
            obs_idx = sorted_idxs[int(-1*i)]
            output_str += observed_column_names[obs_idx] + ' ({0:.4f}), '.format(corr_matrix[latent_idx, obs_idx])
        output_str = output_str[:-2] + '\n'
    return output_str
