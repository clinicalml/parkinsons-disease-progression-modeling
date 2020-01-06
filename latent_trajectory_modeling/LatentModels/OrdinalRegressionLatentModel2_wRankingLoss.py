import numpy as np, pickle, math
import sys, os
DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DIR_PATH)
from Util.visualizer import *

class OrdinalRegressionLatentModel_Longitudinal(object):
    
    def __init__(self, output_num_categories, num_encoder_hidden=0):
        '''
        output_num_categories: list of number of categories for each output. The length of this list is the number of inputs.
        num_encoder_hidden: number of hidden units in encoder, 0 implies encoder is linear
        '''
        assert len(output_num_categories) > 0
        # every feature should have at least 2 categories
        assert np.sum(np.where(np.array(output_num_categories) <= 2, 1, 0)) == 0
        assert num_encoder_hidden >= 0
        np.random.seed(78923)
        self.output_num_categories = output_num_categories
        self.num_encoder_hidden = num_encoder_hidden
        
    def init_parameters(self, sigma_w, sigma_delta):
        '''
        weights are initialized by N(0, sigma_w^2) -> self.weight1 and self.weight2 for the 2 encoder layers
        first threshold for each feature is initialized by N(0, sigma_theta1^2)
        additional thresholds for each feature are initialized by previous threshold + HN(0, sigma_delta^2)
        self.thresholds is a list of lists, each of length 1 less than the number of categories for that output
        '''
        assert sigma_w >= 0
        assert sigma_delta >= 0
        if self.num_encoder_hidden == 0:
            self.weight1 = np.random.normal(loc=0.0, scale=sigma_w, size=(len(self.output_num_categories), 1))
            self.weight2 = None
        else:
            self.weight1 = np.random.normal(loc=0.0, scale=sigma_w, \
                                            size=(len(self.output_num_categories), self.num_encoder_hidden))
            self.weight2 = np.random.normal(loc=0.0, scale=sigma_w, \
                                            size=(self.num_encoder_hidden, 1))
        self.deltas = []
        self.thresholds = []
        for feat_idx in range(len(self.output_num_categories)):
            self.deltas.append(np.absolute(np.random.normal(loc=0.0, scale=sigma_delta, \
                                                            size=self.output_num_categories[feat_idx]-1)))
            self.thresholds.append(np.cumsum(self.deltas[feat_idx]))
    
    def is_data_format_valid(self, data, patient_orderings=None):
        '''
        checks data is # samples x # dimensions and each feature is 0 - (# categories - 1)
        checks patient orderings contains all the indices exactly once
        '''
        if data.shape[1] != len(self.output_num_categories):
            print('Violated 1')
            return False
        if np.sum(np.where(np.isnan(data), 1, 0)) != 0:
            print('Violated 2')
            return False
        if np.sum(np.where(data < 0, 1, 0)) != 0:
            print('Violated 3')
            return False
        if np.sum(data - np.round(data)) != 0: # check is categorical as integers
            for feat_idx in range(len(self.output_num_categories)):
                if np.sum(data[:,feat_idx] - np.round(data[:,feat_idx])) != 0:
                    print('Violated 4 at ' + str(feat_idx))
                    nonzero_idxs = np.nonzero(data[:,feat_idx] - np.round(data[:,feat_idx]))[0]
                    print(data[nonzero_idxs,feat_idx])
            print('Violated 4')
            return False
        for feat_idx in range(len(self.output_num_categories)):
            if np.sum(np.where(data[:,feat_idx] >= self.output_num_categories[feat_idx], 1, 0)) != 0:
                print('Violated 5')
                return False
        if patient_orderings is not None:
            # all indices must occur exactly once in patient orderings
            idxs_in_orderings = set()
            for patno in patient_orderings:
                if len(idxs_in_orderings.intersection(set(patient_orderings[patno]))) != 0:
                    print('Violated 6')
                    return False
                idxs_in_orderings = idxs_in_orderings.union(set(patient_orderings[patno]))
            if len(idxs_in_orderings) != data.shape[0] or max(idxs_in_orderings) != data.shape[0] - 1 \
                or min(idxs_in_orderings) != 0:
                print('Violated 7')
                return False
        return True
        
    def save_model_parameters(self, output_filename):
        '''
        model parameters will be saved in a dictionary
        '''
        param_dict = dict()
        param_dict['weight1'] = self.weight1
        param_dict['weight2'] = self.weight2
        param_dict['deltas'] = self.deltas
        if sys.version_info[0] == 3:
            write_tag = 'wb'
        else:
            write_tag = 'w'
        with open(output_filename, write_tag) as f:
            pickle.dump(param_dict, f)
        
    def load_model_parameters(self, input_filename):
        '''
        input_filename contains the model parameters in a dictionary
        '''
        if sys.version_info[0] == 3:
            read_tag = 'rb'
        else:
            read_tag = 'r'
        with open(input_filename, read_tag) as f:
            param_dict = pickle.load(f)
        assert set(param_dict.keys()) == {'weight1','weight2','deltas'}
        if self.num_encoder_hidden == 0:
            assert param_dict['weight1'].shape == (len(self.output_num_categories), 1)
            assert param_dict['weight2'] is None
        else:
            assert param_dict['weight1'].shape == (len(self.output_num_categories), self.num_encoder_hidden)
            assert param_dict['weight2'].shape == (self.num_encoder_hidden, 1)
        assert len(self.output_num_categories) == len(param_dict['deltas'])
        for feat_idx in range(len(self.output_num_categories)):
            assert self.output_num_categories[feat_idx] == len(param_dict['deltas'][feat_idx])+1
            assert np.sum(np.where(param_dict['deltas'][feat_idx] < 0, 1, 0)) == 0
        self.weight1 = param_dict['weight1']
        self.weight2 = param_dict['weight2']
        self.deltas = param_dict['deltas']
        self.thresholds = []
        for feat_idx in range(len(self.deltas)):
            self.thresholds.append(np.cumsum(self.deltas[feat_idx]))
            
    def sigmoid(self, x):
        return np.exp(x)/(1.+np.exp(x))
    
    def get_total_training_loss(self, data, patient_orderings, output, latent_factors, alpha):
        '''
        data/output: # samples x # dimensions, each feature is 0 - (# categories - 1)
        patient_orderings: dictionary of patient ID to order of indices for that patient
        per-sample all-threshold one-sided square loss
        alpha is weighting of per-pair ranking loss relative to per-sample square loss
        TODO: if total is an issue for features with different numbers of thresholds, weight it appropriately
        must be called after parameters have been initialized or loaded
        '''
        assert self.is_data_format_valid(data, patient_orderings)
        assert self.is_data_format_valid(output)
        assert data.shape[0] == output.shape[0]
        assert latent_factors.shape == (data.shape[0], 1)
        loss = 0.
        for feat_idx in range(len(self.output_num_categories)):
            for thresh_idx in range(len(self.thresholds[feat_idx])):
                feat_onesided_mask = np.where(np.logical_or(np.logical_and(data[:,feat_idx] < thresh_idx + 1, \
                                                                           output[:,feat_idx] >= thresh_idx + 1), \
                                                            np.logical_and(data[:,feat_idx] >= thresh_idx + 1, \
                                                                           output[:,feat_idx] < thresh_idx + 1)), 1, 0)
                loss += np.sum(np.where(feat_onesided_mask == 1, (latent_factors - self.thresholds[feat_idx][thresh_idx])**2, 0))
        loss /= data.shape[0]
        ranking_loss = 0.
        num_pairs = 0
        for patno in patient_orderings:
            for i in range(len(patient_orderings[patno])-1):
                for j in range(i+1, len(patient_orderings[patno])):
                    i_idx = patient_orderings[patno][i]
                    j_idx = patient_orderings[patno][j]
                    ranking_loss -= float(np.log(self.sigmoid(latent_factors[j_idx]-latent_factors[i_idx])))
                    num_pairs += 1
        loss += alpha*ranking_loss/float(num_pairs)
        return loss
    
    def predict(self, data, return_hidden=False):
        '''
        data: # samples x # dimensions, each feature is 0 - (# categories - 1)
        must be called after parameters have been initialized or loaded
        returns latent factors + decoder output for each sample
        use return_hidden if need hidden units in middle of encoder for training backprop (3rd output)
        '''
        assert self.is_data_format_valid(data)
        if self.num_encoder_hidden == 0:
            assert not return_hidden
            latent_factors = np.dot(data, self.weight1)
        else:
            hidden1_before_relu = np.dot(data, self.weight1)
            hidden1 = np.where(hidden1_before_relu > 0, hidden1_before_relu, 0)
            latent_factors = np.dot(hidden1, self.weight2)
        output = self.decode(latent_factors)
        if return_hidden:
            return latent_factors, output, hidden1
        return latent_factors, output
    
    def decode(self, latent_factors):
        '''
        latent_factors: # samples x 1 or # samples
        must be called after arameters have been initialized or loaded
        returns decoder output for each sample
        '''
        assert len(latent_factors.shape) == 1 or (len(latent_factors.shape) == 2 and latent_factors.shape[1] == 1)
        output = np.empty((latent_factors.shape[0], len(self.output_num_categories)))
        for feat_idx in range(output.shape[1]):
            # set min + max first, then update intermediate values
            output[:,feat_idx] = np.where(latent_factors < self.thresholds[feat_idx][0], 0, \
                                          len(self.thresholds[feat_idx])).flatten()
            for category in range(1,len(self.thresholds[feat_idx])):
                output[:,feat_idx] = np.where(np.logical_and(latent_factors.flatten() >= self.thresholds[feat_idx][category-1], \
                                                             latent_factors.flatten() < self.thresholds[feat_idx][category]), \
                                              category, output[:,feat_idx].flatten())
        return output
    
    def fit(self, data, patient_orderings, valid_data, valid_patient_orderings, learn_rate_w, learn_rate_delta, sigma_w, \
            sigma_delta, max_num_iters, conv_threshold, early_stopping_iters, batch_size, alpha, reinit=True):
        '''
        data: # samples x # dimensions, each feature is 0 - (# categories - 1), used for training
        valid_data: same specs as data
        patient_orderings: dictionary mapping patient ids to order of indices in data
        SGD is run until validation reconstruction loss decrease is at most conv_threshold or increases for early_stopping_iters
            or max_num_iters is reached
        if want to continue training from current model parameters, set reinit to False; default will reinitialize parameters
        '''
        assert self.is_data_format_valid(data, patient_orderings)
        assert self.is_data_format_valid(valid_data, valid_patient_orderings)
        assert learn_rate_w >= 0
        assert learn_rate_delta >= 0
        assert sigma_w >= 0
        assert sigma_delta >= 0
        assert max_num_iters >= 0
        assert conv_threshold >= 0
        assert alpha >= 0
        if reinit:
            self.init_parameters(sigma_w, sigma_delta)
        all_latent_factors, all_pred = self.predict(data)
        old_epoch_loss = self.get_total_training_loss(data, patient_orderings, all_pred, all_latent_factors, alpha)
        num_conv_epochs = 0
        patnos = np.array(list(patient_orderings.keys()))
        for epoch in range(max_num_iters):
            print('Training epoch ' + str(epoch) + ' of ' + str(max_num_iters))
            np.random.shuffle(patnos)
            patno_idx = 0
            while patno_idx < len(patnos):
                # form batches by patients here
                batch_idxs = []
                ordered_pairs = []
                num_pairs = 0
                while len(batch_idxs) < batch_size and patno_idx < len(patnos):
                    patno_start_idx = len(batch_idxs)
                    batch_idxs += patient_orderings[patnos[patno_idx]]
                    patno_end_idx = len(batch_idxs)
                    num_pairs += (patno_end_idx - patno_start_idx) * (patno_end_idx - patno_start_idx - 1) /2
                    for i in range(patno_start_idx, patno_end_idx-1):
                        for j in range(i+1, patno_end_idx):
                            ordered_pairs.append((i,j))
                    patno_idx += 1
                if len(batch_idxs) < 0.5*batch_size:
                    break # if last batch is too small, don't use those samples for this epoch
                batch_samples = data[batch_idxs]
                if self.num_encoder_hidden == 0:
                    batch_latent_factors, batch_preds = self.predict(batch_samples)
                else:
                    batch_latent_factors, batch_preds, batch_hidden1s = self.predict(batch_samples, return_hidden=True)
                batch_loss_latent_deriv = np.zeros((batch_samples.shape[0], 1))
                for feat_idx in range(len(self.output_num_categories)):
                    batch_loss_delta_deriv = np.zeros(self.deltas[feat_idx].shape)
                    for thresh_idx in range(len(self.thresholds[feat_idx])):
                        feat_onesided_mask = np.where(np.logical_or(np.logical_and(batch_samples[:,feat_idx] < thresh_idx + 1, \
                                                                                   batch_preds[:,feat_idx] >= thresh_idx + 1), \
                                                                    np.logical_and(batch_samples[:,feat_idx] >= thresh_idx + 1, \
                                                                                   batch_preds[:,feat_idx] < thresh_idx + 1)), \
                                                      1, 0)
                        batch_loss_delta_deriv[:thresh_idx] \
                            += float(np.sum(np.where(feat_onesided_mask ==1, 2*(self.thresholds[feat_idx][thresh_idx] \
                                                                             - batch_latent_factors), 0)))
                        batch_loss_latent_deriv \
                            += np.where(feat_onesided_mask.reshape((-1,1)) == 1, \
                                        2*(batch_latent_factors - self.thresholds[feat_idx][thresh_idx]), 0)
                    #for thresh_idx in range(len(self.thresholds[feat_idx])):
                    #    self.deltas[feat_idx][thresh_idx] \
                    #        -= learn_rate_delta*batch_loss_delta_deriv[thresh_idx] \
                    #            /(float(batch_size)*(len(self.deltas[feat_idx])-thresh_idx))
                    # if normalize by # of thresholds above delta doesn't work well, try line below
                    self.deltas[feat_idx] -= learn_rate_delta*batch_loss_delta_deriv/float(batch_size)
                    self.deltas[feat_idx] = np.absolute(self.deltas[feat_idx])
                    self.thresholds[feat_idx] = np.cumsum(self.deltas[feat_idx])
                # ranking loss
                if self.num_encoder_hidden == 0:
                    # batch_loss_latent_deriv: n x h, batch_samples: n x d, weight1: d x h
                    batch_loss_weight1_deriv = np.dot(batch_samples.T, batch_loss_latent_deriv)/float(batch_size)
                    # ranking loss
                    for pair in ordered_pairs:
                        batch_loss_weight1_deriv \
                            -= alpha*np.dot((batch_samples[pair[1]]-batch_samples[pair[0]]).reshape((-1,1)), \
                                            (1.-self.sigmoid(batch_latent_factors[pair[1]] \
                                                             -batch_latent_factors[pair[0]])).reshape((1,-1))) \
                                /float(num_pairs)
                    self.weight1 -= learn_rate_w*batch_loss_weight1_deriv
                else:
                    # batch_loss_latent_deriv: n x h, batch_hidden1s: n x h1, weight2: h1 x h
                    batch_loss_weight2_deriv = np.dot(batch_hidden1s.T, batch_loss_latent_deriv)/float(batch_size)
                    for pair in ordered_pairs:
                        batch_loss_weight2_deriv \
                            -= alpha*np.dot((batch_hidden1s[pair[1]]-batch_hidden1s[pair[0]]).reshape((-1,1)), \
                                            (1.-self.sigmoid(batch_latent_factors[pair[1]] \
                                                             -batch_latent_factors[pair[0]])).reshape((1,-1))) \
                                /float(num_pairs)
                    self.weight2 -= learn_rate_w*batch_loss_weight2_deriv
                    # batch_loss_latent_deriv: n x h, batch_hidden1s: n x h1, weight2: h1 x h
                    batch_loss_hidden1_deriv = np.dot(batch_loss_latent_deriv, self.weight2.T)
                    relu_masked_batch_loss_hidden1_deriv = np.where(batch_hidden1s > 0, batch_loss_hidden1_deriv, 0)
                    # batch_loss_hidden1_deriv: n x h1, weight1: d x h1, batch_samples: n x d
                    batch_loss_weight1_deriv = np.dot(batch_samples.T, relu_masked_batch_loss_hidden1_deriv)/float(batch_size)
                    for pair in ordered_pairs:
                        # hidden1: 1 x h1, weight2: h1 x h, samples: 1 x d, weight1: d x h1
                        # d x h1 = (d x 1) * ((1 x h) * (h x h1) with 1 x h1 mask on each sample in pair)
                        batch_loss_weight1_deriv \
                            -= alpha*float(1-self.sigmoid(batch_latent_factors[pair[1]]-batch_latent_factors[pair[0]])) \
                                *(np.where(batch_hidden1s[pair[1]].reshape((1,-1)) > 0, \
                                           np.dot(batch_samples[pair[1]].reshape((-1,1)), \
                                                  self.weight2.T.reshape((1,-1))), 0) \
                                 -np.where(batch_hidden1s[pair[0]].reshape((1,-1)) > 0, \
                                           np.dot(batch_samples[pair[0]].reshape((-1,1)), \
                                                  self.weight2.T.reshape((1,-1))), 0)) \
                                /float(num_pairs)
                    self.weight1 -= learn_rate_w*batch_loss_weight1_deriv
            all_latent_factors, all_pred = self.predict(valid_data)
            epoch_loss = self.get_total_training_loss(valid_data, valid_patient_orderings, all_pred, all_latent_factors, alpha)
            if epoch_loss > old_epoch_loss - conv_threshold:
                num_conv_epochs += 1
                if num_conv_epochs >= early_stopping_iters:
                    print('Early stopping at epoch ' + str(epoch))
                    return epoch_loss
            else:
                num_conv_epochs = 0
            old_epoch_loss = epoch_loss
        return epoch_loss
        
    def visualize_thresholds(self, featnames, filename, agg_backend=False):
        '''
        visualization shows a bar plot of the ranges of the latent factor that correspond to each category of each question
        must be called after init or load
        '''
        # for now ok to assume all have 5 categories since only using MDS-UPDRS
        assert np.sum(np.where(np.array(self.output_num_categories) != 5, 1, 0)) == 0
        threshold_dict0 = dict()
        threshold_dict1 = dict()
        threshold_dict2 = dict()
        threshold_dict3 = dict()
        for feat_idx in range(len(self.thresholds)):
            threshold_dict0[featnames[feat_idx]] = self.thresholds[feat_idx][0]
            threshold_dict1[featnames[feat_idx]] = self.thresholds[feat_idx][1]
            threshold_dict2[featnames[feat_idx]] = self.thresholds[feat_idx][2]
            threshold_dict3[featnames[feat_idx]] = self.thresholds[feat_idx][3]
        plot_threshold(threshold_dict0,threshold_dict1,threshold_dict2,threshold_dict3,filename,agg_backend)
        