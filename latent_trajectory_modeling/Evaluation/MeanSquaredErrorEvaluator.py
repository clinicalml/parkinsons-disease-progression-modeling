import numpy as np, pandas as pd, seaborn as sns, sys, os
import matplotlib.pyplot as plt

class MeanSquaredErrorEvaluator(object):
    
    def get_mses(self, true, pred, plot_filedir, question_names, model_name, agg_backend=False):
        '''
        true + pred are numpy arrays: # samples x # features
        question_names and model_name are used for labeling plots
        A. plots distribution of per-patient MSEs to plot_filedir + per_patient_mse_distrib.jpg/.eps
        B. plots a confusion matrix for each question to plot_filedir + question + _confusion_mat.jpg/.eps
        returns 1. dataframe of mses per question with 'Question' and 'MSE' as columns
                2. total MSE
        '''
        assert true.shape == pred.shape
        assert true.shape[1] == len(question_names)
        mses = np.square(true - pred)
        '''
        Plot distribution of per-patient MSEs
        '''
        per_patient_mses = np.mean(mses, axis=0)
        if agg_backend:
            plt.switch_backend('agg')
        plt.clf()
        plt.hist(per_patient_mses, bins=40)
        plt.xlabel('Per-patient MSE')
        plt.ylabel('Frequency')
        plt.title(model_name)
        if sys.version_info[0] == 3:
            image_filetype = '.eps'
        else:
            image_filetype = '.jpg'
        plt.savefig(plot_filedir + 'per_patient_mse_distrib' + image_filetype)
        '''
        Plot confusion matrices for each question. Assuming all questions are 0-4 for now. Modify this code if not.
        '''
        confusion_mat_dir = plot_filedir + 'confusion_mats/'
        if not os.path.isdir(confusion_mat_dir):
            os.makedirs(confusion_mat_dir)
        per_question_mses = np.mean(mses, axis=0)
        for question_idx in range(len(question_names)):
            question = question_names[question_idx]
            question_df = pd.DataFrame(np.vstack((true[:,question_idx], pred[:,question_idx])).T, columns=['True','Pred'])
            '''
            sklearn has a confusion matrix implementation, but re-implemented here
            because we want counts for 0-4 regardless of whether they exist in the data
            following sklearn convention, true is on y-axis and predicted is on x-axis
            top right is highest
            '''
            num_cats = 5
            confusion_mat = np.empty((num_cats,num_cats))
            for i in range(confusion_mat.shape[0]):
                for j in range(confusion_mat.shape[1]):
                    confusion_mat[i,j] = len(question_df.loc[np.logical_and(question_df['True']==num_cats-i-1, \
                                                                            question_df['Pred']==j)])
            confusion_df = pd.DataFrame(confusion_mat, columns=[str(j) for j in range(num_cats)])
            confusion_df['True'] = [str(num_cats-i-1) for i in range(num_cats)]
            confusion_df = confusion_df.set_index('True')
            if agg_backend:
                plt.switch_backend('agg')
            plt.clf()
            sns.heatmap(confusion_df, vmin=0)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(model_name + ' ' + question)
            plt.savefig(confusion_mat_dir + question.replace(' ', '_').replace('/', '_') + '_confusion_mat' + image_filetype)
        per_question_mse_df = pd.DataFrame(question_names, columns=['Question'])
        per_question_mse_df['MSE'] = per_question_mses
        return per_question_mse_df, np.mean(mses)
            
            
            