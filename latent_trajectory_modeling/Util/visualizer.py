import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from bisect import bisect
from random import shuffle

def corr_matrix_heatmap(df, xcol, ycol, 
                        path_to_file=None, agg_backend=False, 
                        xlabels=None, show_ylabels=True, 
                        xlabel_map=None, ylabel_map=None,
                        show_cbar=True):
    """
    df - a dataframe that contains all data
    xcol - the name of the columns in df that we want to put 
           on the x axis of the correlation matrix
    ycol - the name of the columns in df that we want to put 
           on the y axis of the correlation matrix
    """
    if xlabels is not None:
        assert len(xlabels) == len(xcol)
    xcol = sorted(xcol)
    ycol = sorted(ycol)
    corr_matrix = df.corr()
    corr_matrix = corr_matrix.loc[ycol, xcol]
    if agg_backend:
        plt.switch_backend('agg')
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(2*len(xcol), len(ycol)/3.)
    colormap = sns.diverging_palette(250, 10, as_cmap=True)
    if show_cbar:
        cbar_ax = fig.add_axes([.72, .6, .05, .3])
    else:
        cbar_ax = None
    if show_ylabels:
        yticklabels=ycol
        if ylabel_map is not None:
            yticklabels = [ylabel_map[item] for item in ycol]
    else:
        yticklabels=False
    if xlabels is not None:
        columns_map = dict()
        for idx in range(len(xlabels)):
            columns_map[xcol[idx]] = xlabels[idx]
        corr_matrix.rename(columns=columns_map, inplace=True)

    if xlabel_map is not None:
        corr_matrix.rename(columns=xlabel_map, inplace=True)

    main_plot = sns.heatmap(corr_matrix, cmap=colormap, center=0, ax=ax, 
                            xticklabels='auto', yticklabels=yticklabels, \
                            cbar=show_cbar, cbar_ax=cbar_ax, vmin=-1, vmax=1)

    main_plot.tick_params(labelsize=15)
    # main_plot.set_xticklabels(main_plot.get_xticklabels(), rotation=90)

    if show_ylabels:
        left = 0.7
        right = 0.8
    elif show_cbar:
        left = 0.6
        right = 0.7
    else:
        left = 0.65
        right = 0.75
    plt.subplots_adjust(left=left, right=right, bottom=0.15, top=0.99)
    if path_to_file is not None:
        plt.savefig(path_to_file, transparent=True)

def plot_latent_by_time(df, id_col, time_col, latent_col, 
                        num_individual=20, max_time=3,
                        path_to_file=None, agg_backend=False,
                        model_name=None):
    """
    df - a dataframe that contains all data
    id_col - column name that represents the id of each individual
    time_col - column name that represents time
    latent_col - column name that we want to use as a latent factor
    num_individual - number of individuals we want to plot on the graph
    max_time - max time on the x axis
    """
    
    if agg_backend:
        plt.switch_backend('agg')

    fig, ax = plt.subplots(figsize=(16, 16))
    if model_name is not None:
        plt.title("Patient latent factor over time of {}".format(model_name), fontsize=25)
    unique_patno  = list(df.PATNO.value_counts().keys())
    shuffle(unique_patno)
    for patno in unique_patno[:num_individual]:
        sub_df = df[df[id_col] == patno]
        sub_df = sub_df.sort_values(by=[time_col])
        time = list(sub_df[time_col])
        upper_index = bisect(time, max_time)
        if upper_index > 0:
            time = time[:upper_index]
            latent = list(sub_df[latent_col])[:upper_index]
            ax.plot(time, latent, linewidth=5.0)
            ax.set_xlabel("Time since enrollment (years)", fontsize=60)
            ax.set_ylabel("Latent factor", fontsize=60)
            ax.tick_params(labelsize=50)

    if path_to_file is not None:
        plt.tight_layout()
        plt.savefig(path_to_file, transparent=True)

def plot_threshold(threshold_dict0,threshold_dict1,threshold_dict2,threshold_dict3,figurename, agg_backend=False):
    assert set(threshold_dict0.keys()) == set(threshold_dict1.keys())
    assert set(threshold_dict0.keys()) == set(threshold_dict2.keys())
    assert set(threshold_dict0.keys()) == set(threshold_dict3.keys())
    fig, ax = plt.subplots( figsize=(20, 4))
    if agg_backend:
        plt.switch_backend('agg')
    plt.clf()
    feat_names = list(threshold_dict0.keys())
    feat_names.sort()
    threshold0_vals = []
    threshold1_vals = []
    threshold2_vals = []
    threshold3_vals = []
    for feat in feat_names:
        threshold0_vals.append(threshold_dict0[feat])
        threshold1_vals.append(threshold_dict1[feat])
        threshold2_vals.append(threshold_dict2[feat])
        threshold3_vals.append(threshold_dict3[feat])
    assert np.sum(np.where(np.array(threshold1_vals) - np.array(threshold0_vals) < 0, 1, 0)) == 0
    assert np.sum(np.where(np.array(threshold2_vals) - np.array(threshold1_vals) < 0, 1, 0)) == 0
    assert np.sum(np.where(np.array(threshold3_vals) - np.array(threshold2_vals) < 0, 1, 0)) == 0
    
    # plt.bar takes height of threshold rectangle (threshold - previous threshold) as height parameter
    # plt.bar takes previous threshold as bottom parameter
    thresh4_heights = (1.1*max(threshold3_vals) - np.array(threshold3_vals)).tolist()
    plt.bar(range(len(threshold_dict0)), thresh4_heights, bottom=threshold3_vals, align="center", label = '4')
    thresh3_heights = (np.array(threshold3_vals) - np.array(threshold2_vals)).tolist()
    plt.bar(range(len(threshold_dict0)), thresh3_heights, bottom=threshold2_vals, align="center", label = '3')
    thresh2_heights = (np.array(threshold2_vals) - np.array(threshold1_vals)).tolist()
    plt.bar(range(len(threshold_dict0)), thresh2_heights, bottom=threshold1_vals, align="center", label = '2')
    thresh1_heights = (np.array(threshold1_vals) - np.array(threshold0_vals)).tolist()
    plt.bar(range(len(threshold_dict0)), thresh1_heights, bottom=threshold0_vals, align="center", label = '1')
    min_latent_for_plot = min(threshold0_vals)
    if min_latent_for_plot > 0:
        min_latent_for_plot = 0
    else:
        min_latent_for_plot *= 1.1
    thresh0_heights = (np.array(threshold0_vals) - min_latent_for_plot).tolist()
    plt.bar(range(len(threshold_dict0)), thresh0_heights, bottom=len(threshold0_vals)*[min_latent_for_plot], \
            align="center", label = '0')

#     plt.xticks(range(len(threshold_dict0)), [i.strip('_untreated') for i in feat_names])
    plt.xticks(range(len(threshold_dict0)), feat_names)
    plt.xticks(rotation='vertical')
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Latent factor', fontsize=14)
    plt.subplots_adjust(bottom=0.50)
    #plt.yticks(np.arange(0,10,2), ('-4', '-2', '0', '2', '4')) # relabel the threshold 
#     plt.tight_layout()
    plt.savefig(figurename)