import numpy as np, sys, pickle, os, pandas as pd, math
from LatentModels.OrdinalRegressionLatentModel2 import OrdinalRegressionLatentModel
from LatentModels.OrdinalRegressionLatentModel2_wRankingLoss import OrdinalRegressionLatentModel_Longitudinal
from LatentModels.LinearFactorAnalysis import LinearFactorAnalysis
from DataLoaders.PPMI_loader import PPMI_loader
# TODO: import simulation module
from Evaluation.RankingEvaluator import RankingEvaluator
from Evaluation.MeanSquaredErrorEvaluator import MeanSquaredErrorEvaluator
from PostLatentModels.CorrelationCalculator import CorrelationCalculator
from PostLatentModels.LatentRateCalculator import LatentRateCalculator
from PostLatentModels.LatentRateClusterClassifier import LatentRateClusterClassifier

'''
Take in parameters.
'''
param_msg = 'First parameter: PPMI or simulated for data source.\n' \
    + 'Second parameter: path to PPMI data file or simulator settings.\n' \
    + 'Third parameter: LatentFactor, Aging, Ordinal, OrdinalLinear, or OrdinalLongitudinal for model type.\n' \
    + 'Optional fourth to fifth parameters: EvaluationOnly and path to pickle file containing the model parameters for fold 0.' \
    + ' If other folds are available ending in _fold1.pkl through _fold4.pkl, mean + SE will be obtained using these.\n' \
    + 'Optional last parameter: RatePrediction. Fit a linear regression to the latent factors ' \
    + 'and use for predicting future scores and clustering.'
if sys.version_info[0] == 3:
    write_tag = 'wb'
    read_tag = 'rb'
    image_filetype = '.png'
else:
    write_tag = 'w'
    read_tag = 'r'
    image_filetype = '.jpg'
if len(sys.argv) < 4:
    print(param_msg)
    sys.exit()
data_source = sys.argv[1]
if data_source not in {'PPMI', 'simulated'}:
    print(param_msg)
    sys.exit()
path_PPMI_data_dir = sys.argv[2]
model_type = sys.argv[3]
if model_type not in {'LatentFactor', 'Aging', 'Ordinal', 'OrdinalLinear', 'OrdinalLongitudinal'}:
    print(param_msg)
    sys.exit()
eval_only = False
do_rate_pred = False
if len(sys.argv) > 4:
    fourth_param = sys.argv[4]
    if fourth_param not in {'EvaluationOnly', 'RatePrediction'}:
        print(param_msg)
        sys.exit()
    if fourth_param == 'EvaluationOnly':
        eval_only = True
        if len(sys.argv) < 6:
            print(param_msg) # Need pickle file next
            sys.exit()
        path_to_pickle = sys.argv[5]
        assert path_to_pickle.endswith('.pkl')
        with open(path_to_pickle, read_tag) as f:
            eval_model_params = pickle.load(f)
        eval_params_list = [eval_model_params]
        path_to_pickles_list = [path_to_pickle]
        # get multiple pickles if exist
        fold1_path_to_pickle = path_to_pickle[:-4] + '_fold1.pkl'
        if os.path.isfile(fold1_path_to_pickle):
            for idx in range(1,5):
                fold_path_to_pickle = path_to_pickle[:-4] + '_fold' + str(idx) + '.pkl'
                assert os.path.isfile(fold_path_to_pickle)
                with open(fold_path_to_pickle, read_tag) as f:
                    fold_params = pickle.load(f)
                eval_params_list.append(fold_params)
                path_to_pickles_list.append(fold_path_to_pickle)
        if len(sys.argv) >= 7:
            sixth_param = sys.argv[6]
            if sixth_param != 'RatePrediction':
                print(param_msg)
                sys.exit()
            do_rate_pred = True
    else: # RatePrediction
        do_rate_pred = True

'''
Load data
'''
def get_patient_orderings(data):
    # returns data sorted by 'PATNO' then 'EVENT_ID_DUR' + dictionary mapping PATNO to list of ordered indices
    data = data.sort_values(by=['PATNO','EVENT_ID_DUR'])
    patient_orderings = dict()
    patnos = data.PATNO.values
    for idx in range(len(patnos)):
        patno = patnos[idx]
        if patno in patient_orderings:
            patient_orderings[patno].append(idx)
        else:
            patient_orderings[patno] = [idx]
    return data, patient_orderings

if data_source == 'PPMI':
    data_loader = PPMI_loader(os.path.join(path_PPMI_data_dir, "PD_questions_across_time.csv"))
    train_data, valid_data, test_data, observed_column_names = data_loader.get_train_valid_test_split()
    all_train_folds = [train_data]
    all_valid_folds = [valid_data]
    all_test_folds = [test_data]
    for fold in range(4):
        other_fold_train, other_fold_valid, other_fold_test, _ = data_loader.get_train_valid_test_split(fold=fold)
        all_train_folds.append(other_fold_train)
        all_valid_folds.append(other_fold_valid)
        all_test_folds.append(other_fold_test)
    if model_type == 'OrdinalLongitudinal':
        all_train_orderings = []
        all_valid_orderings = []
        for idx in range(len(all_train_folds)):
            all_train_folds[idx], train_patient_orderings = get_patient_orderings(all_train_folds[idx])
            all_train_orderings.append(train_patient_orderings)
            all_valid_folds[idx], valid_patient_orderings = get_patient_orderings(all_valid_folds[idx])
            all_valid_orderings.append(valid_patient_orderings)
    human_readable_dict = data_loader.get_human_readable_dict()
    if do_rate_pred:
        all_train_baseline_folds = []
        all_valid_baseline_folds = []
        all_test_baseline_folds = []
        for idx in range(len(all_train_folds)):
            train_data = all_train_folds[idx]
            valid_data = all_valid_folds[idx]
            test_data = all_test_folds[idx]
            train_valid_test_patnos = dict()
            train_valid_test_patnos['train'] = set(train_data.PATNO.unique().tolist())
            train_valid_test_patnos['valid'] = set(valid_data.PATNO.unique().tolist())
            train_valid_test_patnos['test'] = set(test_data.PATNO.unique().tolist())
            train_baseline, valid_baseline, test_baseline, baseline_column_names \
                = data_loader.get_baseline_data_split(path_PPMI_data_dir + 'PD_selected_baseline.csv', train_valid_test_patnos)
            all_train_baseline_folds.append(train_baseline)
            all_valid_baseline_folds.append(valid_baseline)
            all_test_baseline_folds.append(test_baseline)
        baseline_human_readable_dict = data_loader.get_baseline_human_readable_dict()
else:
    pass

if 'Ordinal' in model_type:
    # round data to integers - some times 2 exams were taken at a visit so the data has averages ending in .5
    for idx in range(len(all_train_folds)):
        all_train_folds[idx][observed_column_names] = all_train_folds[idx][observed_column_names].round()
        all_valid_folds[idx][observed_column_names] = all_valid_folds[idx][observed_column_names].round()
        all_test_folds[idx][observed_column_names] = all_test_folds[idx][observed_column_names].round()
    
parameters_dir = 'final_results/LatentModelParameters/'
if not os.path.isdir(parameters_dir):
    os.makedirs(parameters_dir)

'''
Train model if not in evaluation only mode and get test prediction.
'''
if 'Ordinal' in model_type and not eval_only:
    train_data = all_train_folds[0]
    valid_data = all_valid_folds[0]
    test_data = all_test_folds[0]
    # Run a hyperparameter search.
    num_encoder_hiddens = [5] #[0, 5, 10] #, 20]#, 40]
    learn_rate_ws = [1e-2] #[1e-2, 1e-3, 1e-4]
    #learn_rate_deltas = [5e-3, 5e-4, 5e-5] # instead, set learn_rate_delta to .5*learn_rate_w, o.w. scale way off
    sigma_ws = [1e-3] #[1e-1, 1e-2, 1e-3]
    #sigma_deltas = [5e-2, 5e-3, 5e-4] # instead, set sigma_delta to .5*sigma_w
    max_num_iters = 500
    conv_threshold = 1e-3
    early_stopping_iters = 20
    batch_sizes = [50]#, 200]
    if model_type == 'OrdinalLongitudinal':
        alphas = [1e-1] #[1e-1, 1, 10]
        train_patient_orderings = all_train_orderings[0]
        valid_patient_orderings = all_valid_orderings[0]
    else:
        alphas = [1] # placeholder to ignore
    results_dict = dict() # map tuple of settings to tuple of evaluation metrics
    results_latex_str = 'Enc hidden & lr enc & lr \\theta & \\sigma enc & \\sigma \\Delta & Max iters ' \
        + '& Conv thresh & Early stop iters & Batch size & '
    if model_type == 'OrdinalLongitudinal':
        results_latex_str += 'Alpha & '
    results_latex_str += 'Validation loss \\\\\n\hline\n'
    best_valid_loss = float('inf')
    best_valid_model = None
    best_valid_hyperparams = None
    for num_encoder_hidden in num_encoder_hiddens:
        for learn_rate_w in learn_rate_ws:
            learn_rate_delta = .5*learn_rate_w
            for sigma_w in sigma_ws:
                sigma_delta = .5*sigma_w
                for batch_size in batch_sizes:
                    for alpha in alphas:
                        settings_str = str(num_encoder_hidden) + ' & ' + str(learn_rate_w) + ' & ' \
                            + str(learn_rate_delta) + ' & ' + str(sigma_w) + ' & ' \
                            + str(sigma_delta) + ' & ' + str(max_num_iters) + ' & ' + str(conv_threshold) + ' & ' \
                            + str(early_stopping_iters) + ' & ' + str(batch_size)
                        if model_type == 'OrdinalLongitudinal':
                            settings_str += ' & ' + str(alpha)
                        print('Running ' + settings_str)
                        if model_type in {'Ordinal', 'OrdinalLinear'}:
                            if model_type == 'OrdinalLinear':
                                assert num_encoder_hidden == 0
                            ord_model = OrdinalRegressionLatentModel([5]*len(observed_column_names), \
                                                                     num_encoder_hidden=num_encoder_hidden)
                            valid_loss = ord_model.fit(train_data[observed_column_names].values, \
                                                       valid_data[observed_column_names].values, \
                                                       learn_rate_w, learn_rate_delta, sigma_w, \
                                                       sigma_delta, max_num_iters, conv_threshold, \
                                                       early_stopping_iters, batch_size)
                        else:
                            ord_model = OrdinalRegressionLatentModel_Longitudinal([5]*len(observed_column_names), \
                                                                                 num_encoder_hidden=num_encoder_hidden)
                            valid_loss = ord_model.fit(train_data[observed_column_names].values, \
                                                       train_patient_orderings, \
                                                       valid_data[observed_column_names].values, \
                                                       valid_patient_orderings, \
                                                       learn_rate_w, learn_rate_delta, sigma_w, \
                                                       sigma_delta, max_num_iters, conv_threshold, \
                                                       early_stopping_iters, batch_size, alpha)
                        #print(valid_loss)
                        output_filename = parameters_dir + 'ordinal'
                        if model_type == 'OrdinalLongitudinal':
                            output_filename += 'longitudinal'
                        output_filename += '_' + str(num_encoder_hidden) + 'hidden1_' \
                            + str(learn_rate_w) + 'lrw_' \
                            + str(learn_rate_delta) + 'lrdelta_' \
                            + str(sigma_w) + 'sigmaw_' \
                            + str(sigma_delta) + 'sigmadelta_' \
                            + str(max_num_iters) + 'maxiters_' \
                            + str(conv_threshold) + 'convthresh_' \
                            + str(early_stopping_iters) + 'earlystopiters_' \
                            + str(batch_size) + 'batchsize'
                        if model_type == 'OrdinalLongitudinal':
                            output_filename += '_' + str(alpha) + 'alpha'
                        output_filename += '.pkl'
                        ord_model.save_model_parameters(output_filename)
                        if model_type in {'Ordinal', 'OrdinalLinear'}:
                            settings_key = (num_encoder_hidden, learn_rate_w, learn_rate_delta, sigma_w, sigma_delta, \
                                            max_num_iters, conv_threshold, early_stopping_iters, batch_size)
                        else:
                            settings_key = (num_encoder_hidden, learn_rate_w, learn_rate_delta, sigma_w, sigma_delta, \
                                            max_num_iters, conv_threshold, early_stopping_iters, batch_size, alpha)
                        results_dict[settings_key] = valid_loss
                        results_latex_str += settings_str + ' & ' + str(valid_loss) + '\\\\\n'
                        if not np.isnan(valid_loss) and valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            best_valid_model = ord_model
                            best_valid_hyperparams = settings_key
                            print('Best model so far')
    overall_output_picklefilename = parameters_dir + model_type.lower() + '_hyperparam_search.pkl'
    with open(overall_output_picklefilename, write_tag) as f:
        pickle.dump(results_dict, f)
    overall_output_tablefilename = parameters_dir + model_type.lower() + '_hyperparam_search_table.txt'
    with open(overall_output_tablefilename, 'w') as f:
        f.write(results_latex_str)
    train_latent, train_pred = best_valid_model.predict(train_data[observed_column_names].values)
    valid_latent, valid_pred = best_valid_model.predict(valid_data[observed_column_names].values)
    test_latent, test_pred = best_valid_model.predict(test_data[observed_column_names].values)
    # Use same hyperparameter settings for other folds
    all_models = [best_valid_model]
    all_train_latent_folds = [train_latent]
    all_valid_latent_folds = [valid_latent]
    all_test_latent_folds = [test_latent]
    all_test_pred_folds = [test_pred]
    for idx in range(1, len(all_train_folds)):
        train_data = all_train_folds[idx]
        valid_data = all_valid_folds[idx]
        test_data = all_test_folds[idx]
        if model_type in {'Ordinal', 'OrdinalLinear'}:
            ord_model = OrdinalRegressionLatentModel([5]*len(observed_column_names), \
                                                     num_encoder_hidden=best_valid_hyperparams[0])
            ord_model.fit(train_data[observed_column_names].values, \
                          valid_data[observed_column_names].values, \
                          best_valid_hyperparams[1], best_valid_hyperparams[2], best_valid_hyperparams[3], \
                          best_valid_hyperparams[4], best_valid_hyperparams[5], best_valid_hyperparams[6], \
                          best_valid_hyperparams[7], best_valid_hyperparams[8])
        else:
            train_patient_orderings = all_train_orderings[idx]
            valid_patient_orderings = all_valid_orderings[idx]
            ord_model = OrdinalRegressionLatentModel_Longitudinal([5]*len(observed_column_names), \
                                                                  num_encoder_hidden=best_valid_hyperparams[0])
            ord_model.fit(train_data[observed_column_names].values, \
                          train_patient_orderings, \
                          valid_data[observed_column_names].values, \
                          valid_patient_orderings, \
                          best_valid_hyperparams[1], best_valid_hyperparams[2], best_valid_hyperparams[3], \
                          best_valid_hyperparams[4], best_valid_hyperparams[5], best_valid_hyperparams[6], \
                          best_valid_hyperparams[7], best_valid_hyperparams[8], best_valid_hyperparams[9])
        output_filename = parameters_dir + 'ordinal'
        if model_type == 'OrdinalLongitudinal':
            output_filename += 'longitudinal'
        output_filename += '_' + str(num_encoder_hidden) + 'hidden1_' \
            + str(learn_rate_w) + 'lrw_' \
            + str(learn_rate_delta) + 'lrdelta_' \
            + str(sigma_w) + 'sigmaw_' \
            + str(sigma_delta) + 'sigmadelta_' \
            + str(max_num_iters) + 'maxiters_' \
            + str(conv_threshold) + 'convthresh_' \
            + str(early_stopping_iters) + 'earlystopiters_' \
            + str(batch_size) + 'batchsize'
        if model_type == 'OrdinalLongitudinal':
            output_filename += '_' + str(alpha) + 'alpha'
        output_filename += '_fold' + str(idx) + '.pkl'
        ord_model.save_model_parameters(output_filename)
        all_models.append(ord_model)
        train_latent, train_pred = ord_model.predict(train_data[observed_column_names].values)
        valid_latent, valid_pred = ord_model.predict(valid_data[observed_column_names].values)
        test_latent, test_pred = ord_model.predict(test_data[observed_column_names].values)
        all_train_latent_folds.append(train_latent)
        all_valid_latent_folds.append(valid_latent)
        all_test_latent_folds.append(test_latent)
        all_test_pred_folds.append(test_pred)
elif 'Ordinal' in model_type and eval_only:
    # get num_encoder_hidden from model parameters
    all_models = []
    all_train_latent_folds = []
    all_valid_latent_folds = []
    all_test_latent_folds = []
    all_test_pred_folds = []
    for idx in range(len(eval_params_list)):
        eval_model_params = eval_params_list[idx]
        train_data = all_train_folds[idx]
        valid_data = all_valid_folds[idx]
        test_data = all_test_folds[idx]
        if eval_model_params['weight2'] is None:
            num_encoder_hidden = 0
        else:
            num_encoder_hidden = eval_model_params['weight2'].shape[0]
        if model_type in {'Ordinal', 'OrdinalLinear'}:
            if model_type == 'OrdinalLinear':
                assert num_encoder_hidden == 0
            best_valid_model = OrdinalRegressionLatentModel([5]*len(observed_column_names), \
                                                            num_encoder_hidden=num_encoder_hidden)
        else:
            best_valid_model = OrdinalRegressionLatentModel_Longitudinal([5]*len(observed_column_names), \
                                                                  num_encoder_hidden=num_encoder_hidden)
        best_valid_model.load_model_parameters(path_to_pickle)
        train_latent, train_pred = best_valid_model.predict(train_data[observed_column_names].values)
        valid_latent, valid_pred = best_valid_model.predict(valid_data[observed_column_names].values)
        test_latent, test_pred = best_valid_model.predict(test_data[observed_column_names].values)
        all_models.append(best_valid_model)
        all_train_latent_folds.append(train_latent)
        all_valid_latent_folds.append(valid_latent)
        all_test_latent_folds.append(test_latent)
        all_test_pred_folds.append(test_pred)
elif model_type == 'LatentFactor':
    all_models = []
    all_train_latent_folds = []
    all_valid_latent_folds = []
    all_test_latent_folds = []
    all_test_pred_folds = []
    for idx in range(len(all_train_folds)):
        train_data = all_train_folds[idx]
        valid_data = all_valid_folds[idx]
        test_data = all_test_folds[idx]
        
        linearfa = LinearFactorAnalysis()
        train_valid_arr = np.concatenate((train_data[observed_column_names].values, valid_data[observed_column_names].values))
        linearfa.fit(train_valid_arr)
        train_latent, train_pred = linearfa.predict(train_data[observed_column_names].values)
        valid_latent, valid_pred = linearfa.predict(valid_data[observed_column_names].values)
        test_latent, test_pred = linearfa.predict(test_data[observed_column_names].values)
        all_models.append(linearfa)
        all_train_latent_folds.append(train_latent)
        all_valid_latent_folds.append(valid_latent)
        all_test_latent_folds.append(test_latent)
        all_test_pred_folds.append(test_pred)
else:
    pass

'''
Evaluate model and analyze correlation between latent factors and observed features.
'''
if eval_only:
    eval_output_dir = 'final_results/' + model_type + '_evaluation_redo/'
else:
    eval_output_dir = 'final_results/' + model_type + '_evaluation/'
if not os.path.isdir(eval_output_dir):
    os.makedirs(eval_output_dir)

for idx in range(len(all_test_folds)):
    for latent_idx in range(test_latent.shape[1]):
        all_train_folds[idx]['latent' + str(latent_idx)] = all_train_latent_folds[idx][:,latent_idx]
        all_valid_folds[idx]['latent' + str(latent_idx)] = all_valid_latent_folds[idx][:,latent_idx]
        all_test_folds[idx]['latent' + str(latent_idx)] = all_test_latent_folds[idx][:,latent_idx]
latent_column_names = ['latent' + str(i) for i in range(test_latent.shape[1])]

human_readable_observed_column_names = []
for col in observed_column_names:
    human_readable_observed_column_names.append(human_readable_dict[col])

corr_calculator = CorrelationCalculator()
#if len(latent_column_names) == 1:
    #xlabels = [model_type]
    #xlabels = None
#else:
    #xlabels = [model_type + str(i) for i in range(len(latent_column_names))]
xlabels = [ str(i) for i in range(len(latent_column_names))]
# only show ylabels for LatentFactor since that is the leftmost figure
# only show color bar for OrdinalLongitudinal since that is the rightmost figure
if model_type == 'LatentFactor':
    show_ylabels = True
else:
    show_ylabels = False
if model_type == 'OrdinalLongitudinal':
    show_cbar = True
else:
    show_cbar = False
corr_calculator.calculate_correlations(all_test_folds[0], latent_column_names, observed_column_names, \
                                       human_readable_observed_column_names, \
                                       eval_output_dir + 'latent_obs_corr' + image_filetype, \
                                       eval_output_dir + 'latent_obs_top5corr.pkl', model_type, agg_backend=True, \
                                       xlabels=xlabels, show_ylabels=show_ylabels, show_cbar=show_cbar)

reconstruction_mses = []
consec_visit_rankings_dict = dict() # map latent column name to list
concordance_indices_dict = dict()
for idx in range(len(all_train_folds)):
    test_data = all_test_folds[idx]
    test_latent = all_test_latent_folds[idx]
    test_pred = all_test_pred_folds[idx]
    best_valid_model = all_models[idx]
    
    mse_evaluator = MeanSquaredErrorEvaluator()
    per_question_mses, avg_mse = mse_evaluator.get_mses(test_data[observed_column_names].values, test_pred, eval_output_dir, \
                                                        human_readable_observed_column_names, model_type, agg_backend=True)
    reconstruction_mses.append(avg_mse)
    if idx == 0:
        per_question_mses.to_csv(eval_output_dir + 'per_question_mses.csv', index=False)

    rank_evaluator = RankingEvaluator()
    highest_concordance_index = 0.
    corresponding_consec_visit_ranking = 0.
    for latent_col in latent_column_names:
        factor_col = [latent_col]
        concordance_index = rank_evaluator.get_concordance_index(test_data, factor_col)
        if idx == 0:
            concordance_indices_dict[latent_col] = [concordance_index]
        else:
            concordance_indices_dict[latent_col].append(concordance_index)
        consec_ranking_imagepath = None
        if idx == 0:
            consec_ranking_imagepath = eval_output_dir + latent_col + '_consec_visit_ranking' + image_filetype
        consec_visit_ranking = rank_evaluator.get_consec_visit_ranking(test_data, latent_col, consec_ranking_imagepath, \
                                                                       model_type, agg_backend=True)
        if concordance_index > highest_concordance_index:
            highest_concordance_index = concordance_index
            corresponding_consec_visit_ranking = consec_visit_ranking
        if idx == 0:
            concordance_indices_dict[latent_col] = [concordance_index]
            consec_visit_rankings_dict[latent_col] = [consec_visit_ranking]
            consec_visit_image_filepath = eval_output_dir + latent_col + '_across_time' + image_filetype
            rank_evaluator.plot_latent_distributions_across_time(test_data, latent_col, \
                                                                 consec_visit_image_filepath, 
                                                                 model_type, agg_backend=True)
        else:
            if latent_col in concordance_indices_dict:
                concordance_indices_dict[latent_col].append(concordance_index)
                consec_visit_rankings_dict[latent_col].append(consec_visit_ranking)
            else: # some folds could have fewer latent factors
                concordance_indices_dict[latent_col] = [concordance_index]
                consec_visit_rankings_dict[latent_col] = [consec_visit_ranking]
    if idx == 0:
        concordance_indices_dict['highest'] = [highest_concordance_index]
        consec_visit_rankings_dict['highest'] = [corresponding_consec_visit_ranking]
    else:
        concordance_indices_dict['highest'].append(highest_concordance_index)
        consec_visit_rankings_dict['highest'].append(corresponding_consec_visit_ranking)
    if 'Ordinal' in model_type and idx == 0:
        best_valid_model.visualize_thresholds(human_readable_observed_column_names, \
                                              eval_output_dir + 'thresholds' + image_filetype)
        if model_type != 'OrdinalLinear':
            # Also make heat-maps of correlation between the hidden units and the observed features
            _,_, test_hidden = best_valid_model.predict(test_data[observed_column_names].values, return_hidden=True)
            hidden_column_names = ['hidden' + str(idx) for idx in range(test_hidden.shape[1])]
            xlabels = [str(idx) for idx in range(test_hidden.shape[1])]
            for hidden_idx in range(test_hidden.shape[1]):
                test_data['hidden' + str(hidden_idx)] = test_hidden[:,hidden_idx]
            corr_calculator.calculate_correlations(test_data, hidden_column_names, observed_column_names, \
                                                   human_readable_observed_column_names, \
                                                   eval_output_dir + 'hidden_obs_corr' + image_filetype, \
                                                   eval_output_dir + 'hidden_obs_top5corr.pkl', model_type, agg_backend=True, \
                                                   xlabels=xlabels, show_ylabels=False, show_cbar=False)

reconstruction_mses = np.array(reconstruction_mses)
reconstruction_mean = np.mean(reconstruction_mses)
reconstruction_std = np.std(reconstruction_mses)
output_str = 'Reconstruction MSE: ' + str(reconstruction_mean)
if len(reconstruction_mses) > 1:
    output_str += ' (' + str(reconstruction_std) + ')'
output_str += '\n'
latent_cols_list = list(consec_visit_rankings_dict.keys())
latent_cols_list.sort()
for latent_col in latent_cols_list:
    latent_consec_arr = np.array(consec_visit_rankings_dict[latent_col])
    latent_consec_mean = np.mean(latent_consec_arr)
    latent_consec_std = np.std(latent_consec_arr)
    latent_concord_arr = np.array(concordance_indices_dict[latent_col])
    latent_concord_mean = np.mean(latent_concord_arr)
    latent_concord_std = np.std(latent_concord_arr)
    output_str += latent_col + ' consecutive visit ranking: ' + str(latent_consec_mean) 
    if len(latent_consec_arr) > 1:
        output_str += ' (' + str(latent_consec_std) + ')'
    output_str += '\n'
    output_str += latent_col + ' concordance index: ' + str(latent_concord_mean) 
    if len(latent_concord_arr) > 1:
        output_str += ' (' + str(latent_concord_std) + ')'
    output_str += '\n'
with open(eval_output_dir + 'eval_metrics.txt', 'w') as f:
    f.write(output_str)
                                                                                              
'''
If rate prediction is specified:
- calculate rate of latent factors for all patients with at least 2 data points
- divide into negative, around 0, and positive (maybe separating large positive) rates
- predict cluster using baseline features
- print a table of cluster means for significant features in baseline classifier
- individual linear regressions of latent factor (diffs w/ 1st bullet: use only first 50% of points of test patients with at least 3 data points, also calculates intercept)
- MSE of predicting using extrapolated latent factors
'''
if not do_rate_pred:
    sys.exit() # done
rate_pred_dir = 'final_results/' + model_type + '_RatePrediction/'
if not os.path.isdir(rate_pred_dir):
    os.makedirs(rate_pred_dir)

extrapolate_mses = []
cluster_metrics_dict = dict() # map metric name to list
for idx in range(len(all_train_folds)):
    train_data = all_train_folds[idx]
    valid_data = all_valid_folds[idx]
    test_data = all_test_folds[idx]
    test_latent = all_test_latent_folds[idx]
    test_pred = all_test_pred_folds[idx]
    best_valid_model = all_models[idx]
    train_baseline_df = all_train_baseline_folds[idx]
    valid_baseline_df = all_valid_baseline_folds[idx]
    test_baseline_df = all_test_baseline_folds[idx]
      
    # calculate latent rates + biases
    rate_calculator = LatentRateCalculator()
    train_data_atleast_2timepoints = rate_calculator.get_atleast_2timepoints_patnos_dataframe(train_data)
    valid_data_atleast_2timepoints = rate_calculator.get_atleast_2timepoints_patnos_dataframe(valid_data)
    test_data_atleast_2timepoints = rate_calculator.get_atleast_2timepoints_patnos_dataframe(test_data)

    train_latent_rates_biases_from_all_timepoints \
        = rate_calculator.get_rate_of_latent(train_data_atleast_2timepoints[['PATNO','EVENT_ID_DUR']+latent_column_names], \
                                             latent_column_names)
    valid_latent_rates_biases_from_all_timepoints \
        = rate_calculator.get_rate_of_latent(valid_data_atleast_2timepoints[['PATNO','EVENT_ID_DUR']+latent_column_names], \
                                             latent_column_names)
    test_latent_rates_biases_from_all_timepoints \
        = rate_calculator.get_rate_of_latent(test_data_atleast_2timepoints[['PATNO','EVENT_ID_DUR']+latent_column_names], \
                                             latent_column_names)
    if idx == 0:
        # plot 20 test samples
        for latent_col in latent_column_names:
            rate_calculator.plot_latent_factors(test_data_atleast_2timepoints[['PATNO','EVENT_ID_DUR',latent_col]], \
                                                latent_col, rate_pred_dir + latent_col + '_across_time' + image_filetype, \
                                                agg_backend=True)

    # cluster by rates
    rate_cluster_classifier = LatentRateClusterClassifier()
    latent_rate_column_names = ['latent' + str(i) + '_rate' for i in range(test_latent.shape[1])]
    train_valid_latent_rates_biases_from_all_timepoints \
         = train_latent_rates_biases_from_all_timepoints.append(valid_latent_rates_biases_from_all_timepoints)
    rate_cluster_classifier.find_quantiles_rate_of_latent(train_valid_latent_rates_biases_from_all_timepoints, 'latent0_rate', 2)
    train_latent_rates_biases_clusters \
        = rate_cluster_classifier.split_by_preset_divisions(train_latent_rates_biases_from_all_timepoints, 'latent0_rate')
    valid_latent_rates_biases_clusters \
        = rate_cluster_classifier.split_by_preset_divisions(valid_latent_rates_biases_from_all_timepoints, 'latent0_rate')
    test_latent_rates_biases_clusters \
        = rate_cluster_classifier.split_by_preset_divisions(test_latent_rates_biases_from_all_timepoints, 'latent0_rate')
    
    # merge clusters with baseline features
    train_clusters_baseline = train_latent_rates_biases_clusters.merge(train_baseline_df, on=['PATNO'], validate='one_to_one')
    valid_clusters_baseline = valid_latent_rates_biases_clusters.merge(valid_baseline_df, on=['PATNO'], validate='one_to_one')
    test_clusters_baseline = test_latent_rates_biases_clusters.merge(test_baseline_df, on=['PATNO'], validate='one_to_one')
    num_test_cluster1 = test_latent_rates_biases_clusters.cluster.sum()
    if num_test_cluster1 == 0 or num_test_cluster1 == len(test_latent_rates_biases_clusters):
        print('Only 1 cluster in test for fold ' + str(idx))
        # No point in classifying clusters for this fold
    else:
        if len(cluster_metrics_dict) == 0:
            # get table of means for clusters in all 3 sets
            train_valid_test_clusters_baseline = pd.concat((train_clusters_baseline, valid_clusters_baseline, \
                                                            test_clusters_baseline))
            rate_cluster_classifier.save_table_of_means_per_cluster(train_valid_test_clusters_baseline,'cluster',\
                                                                    latent_rate_column_names+baseline_column_names, \
                                                                    rate_pred_dir+'cluster_feats_table.txt', \
                                                                    baseline_human_readable_dict)

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
        metrics_filepath = None
        confusion_mat_filepath = None
        if len(cluster_metrics_dict) == 0:
            rate_cluster_classifier.print_top_coeffs(rate_pred_dir + 'classifier_coeffs.txt', baseline_human_readable_dict)
            metrics_filepath = rate_pred_dir + 'cluster_metrics.txt'
            confusion_mat_dir = rate_pred_dir  + 'confusion_mats/'
            if not os.path.isdir(confusion_mat_dir):
                os.makedirs(confusion_mat_dir)
            confusion_mat_filepath = confusion_mat_dir + 'cluster_confusion_mat' + image_filetype

        fold_metrics_dict = rate_cluster_classifier.get_cluster_metrics(test_clusters_baseline[baseline_column_names].values, \
                                                                        test_clusters_baseline['cluster'].values, \
                                                                        metrics_filepath, confusion_mat_filepath, model_type)
        if len(cluster_metrics_dict) == 0:
            for metric in fold_metrics_dict.keys():
                cluster_metrics_dict[metric] = [fold_metrics_dict[metric]]
        else:
            for metric in fold_metrics_dict.keys():
                cluster_metrics_dict[metric].append(fold_metrics_dict[metric])

    # split test data into first half of timepoints and second half of timepoints for prediction
    # note: if interpolation was done, don't count those points in split here
    test_patno_counts = dict(test_data.PATNO.value_counts())
    test_patnos_atleast_3timepoints = set()
    for patno in test_patno_counts.keys():
        if test_patno_counts[patno] >= 3:
            test_patnos_atleast_3timepoints.add(patno)
    test_data_atleast_3timepoints = test_data.loc[test_data['PATNO'].isin(test_patnos_atleast_3timepoints)]
    test_data_first_half_patnos = []
    test_data_first_half_event_id_durs = []
    test_data_second_half_patnos = []
    test_data_second_half_event_id_durs = []
    for patno in test_patnos_atleast_3timepoints:
        patno_df = test_data.loc[test_data['PATNO']==patno]
        patno_df = patno_df.sort_values(by=['EVENT_ID_DUR'])
        first_half_end_idx = int(math.ceil(len(patno_df)/2.))
        test_data_first_half_patnos += first_half_end_idx*[patno]
        test_data_first_half_event_id_durs += patno_df.EVENT_ID_DUR.values[:first_half_end_idx].tolist()
        test_data_second_half_patnos += (len(patno_df)-first_half_end_idx)*[patno]
        test_data_second_half_event_id_durs += patno_df.EVENT_ID_DUR.values[first_half_end_idx:].tolist()
    test_data_first_half = pd.DataFrame({'PATNO': test_data_first_half_patnos, \
                                         'EVENT_ID_DUR': test_data_first_half_event_id_durs})
    test_data_first_half_latents = test_data_first_half.merge(test_data[['PATNO', 'EVENT_ID_DUR']+latent_column_names], \
                                                              validate='one_to_one')

    test_data_first_half_rates_biases = rate_calculator.get_rate_of_latent(test_data_first_half_latents, latent_column_names)
    test_data_second_half = pd.DataFrame({'PATNO': test_data_second_half_patnos, \
                                          'EVENT_ID_DUR': test_data_second_half_event_id_durs})
    test_data_second_half_observeds = test_data_second_half.merge(test_data[['PATNO', 'EVENT_ID_DUR']+observed_column_names], \
                                                                  validate='one_to_one')
    # extrapolate, decode, and get MSE
    test_data_second_half_latents = rate_calculator.extrapolate_latent_factors(test_data_first_half_rates_biases, \
                                                                               test_data_second_half, latent_column_names)
    test_data_second_half_preds = best_valid_model.decode(test_data_second_half_latents[latent_column_names].values)
    second_half_preds_per_question_mses, second_half_preds_avg_mse \
        = mse_evaluator.get_mses(test_data_second_half_observeds[observed_column_names].values, test_data_second_half_preds, \
                                 rate_pred_dir, human_readable_observed_column_names, model_type)
    extrapolate_mses.append(second_half_preds_avg_mse)
    if idx == 0:
        per_question_mses.to_csv(rate_pred_dir + 'per_question_mses.csv', index=False)

extrapolate_mses = np.array(extrapolate_mses)
extrapolate_mean = np.mean(extrapolate_mses)
extrapolate_std = np.std(extrapolate_mses)
output_str = 'Prediction MSE: ' + str(extrapolate_mean) 
if len(extrapolate_mses) > 1:
    output_str += ' (' + str(extrapolate_std) + ')'
output_str += '\n'
output_str += 'Classifying clusters metrics:\n'
metrics_list = list(cluster_metrics_dict.keys())
metrics_list.sort()
for metric in metrics_list:
    metric_arr = np.array(cluster_metrics_dict[metric])
    metric_mean = np.mean(metric_arr)
    metric_std = np.std(metric_arr)
    output_str += metric + ': ' + str(metric_mean) 
    if len(metric_arr) > 1:
        output_str += ' (' + str(metric_std) + ')'
    output_str += '\n'
with open(rate_pred_dir + 'Prediction_Clustering_Metrics.txt', 'w') as f:
    f.write(output_str)
