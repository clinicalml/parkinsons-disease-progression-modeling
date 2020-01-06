import numpy as np, pandas as pd, seaborn as sns, matplotlib, pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

'''
Clusters with kmeans for multiple latent factors or divides into quantiles for single latent factor
Predicts cluster membership
'''

class LatentRateClusterClassifier(object):

    def kmeans_rate_of_latent(self,dataFrame,latent_rate_column_names, num_cluster):
        '''
        Fit clusters on rate of latent factor by Kmeans
        '''
        kmeans = KMeans(n_clusters= num_cluster)
        col_set = {'PATNO'} | set(latent_rate_column_names)
        assert col_set.issubset(set(dataFrame.columns.values.tolist()))
        assert len(dataFrame.dropna(subset=['PATNO']+latent_rate_column_names)) == len(dataFrame)

        points = np.array(dataFrame[latent_rate_column_names])
        km = kmeans.fit(points)
        return km.labels_
    
    def find_quantiles_rate_of_latent(self, dataFrame,latent_rate_column_name, num_quantiles):
        '''
        Fit clusters on rate of latent factor by quantiles
        Only input one column and output an array of quantiles
        '''
        assert num_quantiles > 0
        assert round(num_quantiles) - num_quantiles == 0 # is int
        assert latent_rate_column_name in dataFrame.columns.values.tolist()
        latent_col = dataFrame[latent_rate_column_name]
        self.quantiles = [latent_col.quantile(q=i/float(num_quantiles)) for i in range(1, num_quantiles)]
        return self.quantiles
    
    def split_by_preset_divisions(self, dataFrame, latent_rate_column_name, quantile_splitpoints=None):
        '''
        Must be called after find_quantiles_rate_of_latent if quantile_splitpoints is None
        Returns dataframe with additional column 'cluster' with values 0 through len(quantile_splitpoints)+1
        '''
        if quantile_splitpoints is not None:
            assert len(quantile_splitpoints) > 0
        else:
            quantile_splitpoints = self.quantiles
        assert latent_rate_column_name in dataFrame.columns.values.tolist()
        dataFrame['cluster'] = np.where(dataFrame[latent_rate_column_name] <= quantile_splitpoints[0], 0, \
                                        len(quantile_splitpoints))
        for idx in range(1,len(quantile_splitpoints)):
            dataFrame['cluster'] = np.where(np.logical_and(dataFrame[latent_rate_column_name] > quantile_splitpoints[idx], \
                                                           dataFrame[latent_rate_column_name] <= quantile_splitpoints[idx+1]), \
                                            idx, dataFrame['cluster'])
        return dataFrame
    
    def fit_rate_clusters_classifier(self, train_df, valid_df, cluster_col, baseline_cols):
        '''
        Classify patients into clusters based on baseline features
        '''
        assert cluster_col in train_df.columns.values.tolist()
        assert cluster_col in valid_df.columns.values.tolist()
        assert set(baseline_cols).issubset(set(train_df.columns.values.tolist()))
        assert set(baseline_cols).issubset(set(valid_df.columns.values.tolist()))
        self.cluster_labels = train_df[cluster_col].unique()
        X_train = train_df[baseline_cols] 
        y_train = train_df[cluster_col] 
        X_valid = valid_df[baseline_cols]
        y_valid = valid_df[cluster_col]
        print(X_train.shape, y_train.shape, X_valid.shape,y_valid.shape)
        Cs = [0.1, 0.5, 1, 5 ,10, 20]
        best_classifier = None
        best_valid_metric = 0. # use validation auroc to pick model
        for c in Cs:
            # print(c)
            clf = LogisticRegression(C=c, penalty='l1', solver='saga', max_iter=10000)
            clf.fit(X_train,y_train)
            print(c, clf.coef_)
            valid_prob = clf.predict_proba(X_valid)
            valid_metric = metrics.roc_auc_score(y_valid, valid_prob[:,1])
            if not np.isnan(valid_metric) and valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                best_classifier = clf
        self.baseline_cols = baseline_cols
        self.classifier = best_classifier
        
    def print_top_coeffs(self, coeff_filepath, baseline_human_readable_dict, num_min_max_to_print=10):
        '''
        must be called after fit_rate_clusters_classifier above
        '''
        assert num_min_max_to_print > 0
        assert num_min_max_to_print < len(self.classifier.coef_[0])
        coeffs = self.classifier.coef_[0]
        max_10_idxs = np.argpartition(coeffs, range(int(-1*num_min_max_to_print),0))[int(-1*num_min_max_to_print):]
        min_10_idxs = np.argpartition(coeffs, range(num_min_max_to_print))[:num_min_max_to_print]
        output_str = 'Largest '+ str(num_min_max_to_print) + ' coefficients:\n'
        for idx in max_10_idxs:
            col = self.baseline_cols[idx]
            if col in baseline_human_readable_dict.keys():
                col = baseline_human_readable_dict[col]
            output_str += col + ': ' + str(coeffs[idx]) + '\n'
        output_str += 'Smallest ' + str(num_min_max_to_print) + ' coefficients:\n'
        for idx in min_10_idxs:
            col = self.baseline_cols[idx]
            if col in baseline_human_readable_dict.keys():
                col = baseline_human_readable_dict[col]
            output_str += col + ': ' + str(coeffs[idx]) + '\n'
        with open(coeff_filepath, 'w') as f:
            f.write(output_str)

    def classify_points(self, X):
        '''
        Must be called after fit_rate_clusters_classifier
        '''
        return self.classifier.predict(X)
        
    def classify_points_prob(self, X):
        '''
        Must be called after fit_rate_clusters_classifier
        '''
        return self.classifier.predict_proba(X) 
        
    def get_cluster_metrics(self, X, y_true, metrics_filepath, confusion_mat_filepath, model_name, agg_backend=False):
        '''
        Must be called after fit_rate_clusters_classifier
        '''
        assert X.shape[0] == y_true.shape[0]
        assert X.shape[1] == len(self.classifier.coef_[0])
        assert set(np.unique(y_true).tolist()).issubset(set(self.cluster_labels.tolist()))
        # AUROC, accuracy, precision, recall, F-score, micro-average, macro-average, confusion matrix
        y_proba = self.classifier.predict_proba(X)
        y_pred = self.classifier.predict(X)
        
        auroc = metrics.roc_auc_score(y_true, y_proba[:,1])
        acc = metrics.accuracy_score(y_true, y_pred)
        per_class_precisions = metrics.precision_score(y_true, y_pred, average=None)
        micro_precision = metrics.precision_score(y_true, y_pred, average='micro')
        macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
        per_class_recalls = metrics.recall_score(y_true, y_pred, average=None)
        micro_recall = metrics.recall_score(y_true, y_pred, average='micro')
        macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
        per_class_f1s = metrics.f1_score(y_true, y_pred, average=None)
        micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
        macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        all_metrics_dict = dict()
        all_metrics_dict['AUROC'] = auroc
        all_metrics_dict['Accuracy'] = acc
        for idx in range(len(per_class_precisions)):
            all_metrics_dict['Precision for cluster ' + str(idx)] = per_class_precisions[idx]
        all_metrics_dict['Micro precision'] = micro_precision
        all_metrics_dict['Macro precision'] = macro_precision
        for idx in range(len(per_class_recalls)):
            all_metrics_dict['Recall for cluster ' + str(idx)] = per_class_recalls[idx]
        all_metrics_dict['Micro recall'] = micro_recall
        all_metrics_dict['Macro recall'] = macro_recall
        for idx in range(len(per_class_f1s)):
            all_metrics_dict['F1 score for cluster ' + str(idx)] = per_class_f1s[idx]
        all_metrics_dict['Micro F1 score'] = micro_f1
        all_metrics_dict['Macro F1 score'] = macro_f1
        if metrics_filepath is not None:
            output_str = 'Classifying clusters of latent factor rates from ' + model_name + '\n'
            metrics_list = list(all_metrics_dict.keys())
            metrics_list.sort()
            for metric in metrics_list:
                output_str += metric + ': ' + str(all_metrics_dict[metric]) + '\n'
            with open(metrics_filepath, 'w') as f:
                f.write(output_str)
        
        if confusion_mat_filepath is not None:
            # plot a confusion matrix
            num_clusters = len(self.cluster_labels)
            confusion_mat = np.empty((num_clusters, num_clusters))
            y_true_pred_df = pd.DataFrame({'True': y_true, 'Pred': y_pred})
            for i in range(confusion_mat.shape[0]):
                for j in range(confusion_mat.shape[1]):
                    confusion_mat[i,j] = len(y_true_pred_df.loc[np.logical_and(y_true_pred_df['True']==num_clusters-i-1, \
                                                                               y_true_pred_df['Pred']==j)])
            confusion_df = pd.DataFrame(confusion_mat, columns=[str(j) for j in range(num_clusters)])
            confusion_df['True'] = [str(num_clusters-i-1) for i in range(num_clusters)]
            confusion_df = confusion_df.set_index('True')
            if agg_backend:
                plt.switch_backend('agg')
            plt.clf()
            sns.heatmap(confusion_df, vmin=0)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Classifying clusters of latent rates from ' + model_name)
            plt.savefig(confusion_mat_filepath)
        
        return all_metrics_dict
    
    def save_table_of_means_per_cluster(self, df, cluster_col, feat_cols, filepath, baseline_human_readable_dict):
        '''
        Print the following table:
        feat_name & cluster1_mean & cluster2_mean & cluster3_mean ... & pval \\ 
        \hline
        etc.
        Always print tremor- vs postural instability-dominant, even if insignificant, label w/ * if significant
        TODO: in current implementation, the printed p-value is for the first significantly different pair of clusters,
              so it only makes sense for the 2-cluster setting
        '''
        table_str = 'Feature '
        clusters = sorted(df[cluster_col].unique().tolist())
        for cluster in clusters:
            table_str += '& Cluster ' + str(cluster) + ' mean '
        table_str += ' & p-value \\\\\n\\hline\n'
        for feat in feat_cols:
            significant = False
            for i in range(len(clusters)-1):
                for j in range(i, len(clusters)):
                    cluster_i_feat = df.loc[df[cluster_col]==clusters[i]][feat].values
                    cluster_j_feat = df.loc[df[cluster_col]==clusters[j]][feat].values
                    _, pval = ttest_ind(cluster_i_feat, cluster_j_feat, equal_var=False)
                    if pval < 0.05:
                        significant = True
                        break
                if significant:
                    break
            if significant or feat.startswith('TD_PIGD_untreated'):
                if feat in baseline_human_readable_dict.keys():
                    table_str += baseline_human_readable_dict[feat]
                else:
                    table_str += feat
                if feat.startswith('TD_PIGD_untreated') and significant:
                    table_str += '*'
                for cluster in clusters:
                    table_str += ' & ' + str(df.loc[df[cluster_col]==cluster][feat].mean()) + ' '
                table_str += ' & ' + str(pval) + '\\\\\n\\hline\n'
        with open(filepath, 'w') as f:
            f.write(table_str)