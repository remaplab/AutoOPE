import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame

from common.constants import FIG_EXTENSION
from common.evaluation.evaluation_utils import get_legend_label

import shutil

sns.set_style('whitegrid')
#sns.set_palette('Set2', 5)
plt.rcParams["text.usetex"] = True if shutil.which('latex') else False
# Define all possible methods (based on your usage of get_legend_label)
ALL_METHODS = ['PAS-IF', 'SLOPE', 'OCV IPS', 'OCV DR', 'AutoOPE', 'empty', 'OPERA']

# Assign a fixed color to each method using seaborn or matplotlib colors
PALETTE = dict(zip(ALL_METHODS, sns.color_palette("tab10", n_colors=len(ALL_METHODS))))



def save_figure(es_eval_list, dir_name, metric_name, pi_e, y_label, title="",
                x_label=""):
    es_all_policy_df = pd.DataFrame(index=[], columns=['Methods', 'policy_name', x_label, y_label])
    all_methods_es_df = pd.DataFrame(index=[], columns=['Methods', 'policy_name', x_label, y_label])

    for method, es_eval in es_eval_list:
        if es_eval:
            for policy_name, info in pi_e.items():
                es_single_policy_df = pd.DataFrame({
                    'Methods': get_legend_label(method), 'policy_name': policy_name, x_label: info[1],
                    y_label: es_eval[policy_name][metric_name].values
                })
                es_all_policy_df = pd.concat([es_all_policy_df, es_single_policy_df])

            # Single Estimator Selection Method Plots
            es_all_policy_df = es_all_policy_df.reset_index(drop=True)
            file_path = 'estimator_selection_' + method + '_' + metric_name
            eval_plot_and_save_plain_data(dir_name, file_path, es_all_policy_df, y_label, title, x_label, 'Methods')

            all_methods_es_df = pd.concat([all_methods_es_df, es_all_policy_df])
            es_all_policy_df = pd.DataFrame(index=[], columns=['Methods', 'policy_name', x_label, y_label])

    # All Estimator Selection Methods Plots
    all_methods_es_df = all_methods_es_df.reset_index(drop=True)
    file_path = 'estimator_selection_' + metric_name
    eval_plot_and_save_plain_data(dir_name, file_path, all_methods_es_df, y_label, title, x_label, 'Methods')

    # Pasif and Black-Box
    #if pasif_es_eval and bb_es_eval:
    #    pasif_bb_df = all_methods_es_df[np.logical_or(all_methods_es_df['Methods'] == 'PAS-IF', all_methods_es_df['Methods'] == 'AutoOPE')]
    #    pasif_bb_df = pasif_bb_df.reset_index(drop=True)
    #    file_path = 'estimator_selection_' + metric_name + '_pasif_black-box'
    #    eval_plot_and_save_plain_data(dir_name, file_path, pasif_bb_df, y_label, title, x_label, 'Methods')
#
    ## Const and Black-Box
    #if const_es_val and bb_es_eval:
    #    const_bb_df = all_methods_es_df[np.logical_or(all_methods_es_df['Methods'] == 'SNIPS', all_methods_es_df['Methods'] == 'AutoOPE')]
    #    const_bb_df = const_bb_df.reset_index(drop=True)
    #    file_path = 'estimator_selection_' + metric_name + '_constant_black-box'
    #    eval_plot_and_save_plain_data(dir_name, file_path, const_bb_df, y_label, title, x_label, 'Methods')


def save_subplots_metrics(es_eval_list, dir_name, metric_names, pi_e, y_labels, title="",
                          x_label=""):
    es_all_policy_df = pd.DataFrame(index=[], columns=['Methods', 'policy_name', x_label] + y_labels)
    all_methods_es_df = pd.DataFrame(index=[], columns=['Methods', 'policy_name', x_label] + y_labels)

    for method, es_eval in es_eval_list:
        if es_eval:
            for policy_name, info in pi_e.items():
                es_single_policy_df = {key: es_eval[policy_name][val].values for key, val in zip(y_labels, metric_names)}
                es_single_policy_df['Methods'] = get_legend_label(method)
                es_single_policy_df['policy_name'] = policy_name
                es_single_policy_df[x_label] = info[1]
                es_single_policy_df = pd.DataFrame(es_single_policy_df)
                es_all_policy_df = pd.concat([es_all_policy_df, es_single_policy_df])

            # Single Estimator Selection Method Plots
            es_all_policy_df = es_all_policy_df.reset_index(drop=True)
            all_methods_es_df = pd.concat([all_methods_es_df, es_all_policy_df])
            es_all_policy_df = pd.DataFrame(index=[], columns=['Methods', 'policy_name', x_label] + y_labels)

    # All Estimator Selection Methods Plots
    all_methods_es_df = all_methods_es_df.reset_index(drop=True)
    file_path = 'estimator_selection_subplots'
    eval_subplots_and_save_plain_data(dir_name, file_path, all_methods_es_df, y_labels, title, x_label, 'Methods')

    # Pasif and Black-Box
    #if pasif_es_eval and bb_es_eval:
    #    pasif_bb_df = all_methods_es_df[np.logical_or(all_methods_es_df['Methods'] == 'PAS-IF', all_methods_es_df['Methods'] == 'AutoOPE')]
    #    pasif_bb_df = pasif_bb_df.reset_index(drop=True)
    #    file_path = 'estimator_selection_subplots_pasif_black-box'
    #    eval_subplots_and_save_plain_data(dir_name, file_path, pasif_bb_df, y_labels, title, x_label, 'Methods')


def save_es_figures(evaluation_of_selection_method, log_dir_path, pi_e, x_label, title):
    es_eval_list = evaluation_of_selection_method.get_all_evaluation_results_of_estimator_selection()
    save_figure(es_eval_list, dir_name=log_dir_path, metric_name='relative_regret', pi_e=pi_e, x_label=x_label,
                title=title, y_label='Relative Regret')
    save_figure(es_eval_list, dir_name=log_dir_path, metric_name='rank_correlation_coefficient', pi_e=pi_e,
                x_label=x_label, title=title, y_label='Spearman\'s Rank Correlation')
    save_figure(es_eval_list, dir_name=log_dir_path, metric_name='mse', pi_e=pi_e,
                x_label=x_label, title=title, y_label='MSE')
    save_subplots_metrics(es_eval_list, dir_name=log_dir_path,
                          metric_names=['rank_correlation_coefficient', 'relative_regret'], pi_e=pi_e, x_label=x_label,
                          title=title, y_labels=['Spearman\'s Rank Corr.', 'Relative Regret'])
    save_subplots_metrics(es_eval_list, dir_name=log_dir_path,
                          metric_names=['rank_correlation_coefficient', 'mse'], pi_e=pi_e, x_label=x_label,
                          title=title, y_labels=['Spearman\'s Rank Corr.', 'MSE'])


def save_legend(ax, dir_name, file_path):
    # Utility function to save the legend separately as horizontal
    legend = ax.get_legend()
    if legend is not None:
        legend_fig = plt.figure(figsize=(6, 0.2))  # Adjust width for horizontal layout
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis('off')
        handles, labels = ax.get_legend_handles_labels()
        legend_ax.legend(
            handles,
            labels,
            loc='center',
            frameon=False,
            ncol=len(labels)  # Arrange legend items horizontally
        )
        os.makedirs(dir_name, exist_ok=True)
        legend_path = os.path.join(dir_name, file_path)
        legend_fig.savefig(legend_path, bbox_inches='tight')
        plt.close(legend_fig)


def eval_plot_and_save_plain_data(dir_name, file_path, eval_df, y_label, title, x_label, hue=None):
    if eval_df['policy_name'].unique().shape[0] > 1:
        plt.figure(figsize=(6, 5))
        ax = sns.lineplot(x=x_label, y=y_label, hue=hue, data=eval_df, linewidth=6, alpha=0.75, palette=PALETTE)  # default 95% C.I.
        ax.set_title(title, fontsize=24)
        ax.set_xlabel(x_label, fontsize=24)
        ax.set_ylabel(y_label, fontsize=24)
        if y_label == "Relative Regret" and eval_df[y_label].max() > 10:
            ax.set_yscale('symlog')
        if y_label == "MSE":
            ax.set_yscale('log')
        ax.tick_params(labelsize=30)

        _save_legend = eval_df[hue].unique().shape[0] > 2
        if _save_legend:
            # Save the legend separately
            save_legend(ax, dir_name, 'legend.pdf')

            # Remove the legend from the plot
            ax.legend_.remove()

        else:
            ax.legend(fontsize=5)

        plt.tight_layout()
        save_and_show_plot(ax.get_figure(), dir_name, file_path, False)
    eval_df.to_csv(os.path.join(dir_name, file_path + '.csv'))


def eval_subplots_and_save_plain_data(dir_name, file_path, eval_df, y_labels, title, x_label, hue=None):
    if eval_df['policy_name'].unique().shape[0] > 1:
        fig, ax = plt.subplots(1, len(y_labels), figsize=(8, 4))
        for idx, y_label in enumerate(y_labels):
            ax_i = sns.lineplot(x=x_label, y=y_label, hue=hue, data=eval_df, linewidth=6, alpha=0.75, ax=ax[idx], palette=PALETTE)  # default 95% C.I.
            ax_i.set_xlabel(x_label, fontsize=20)
            ax_i.set_ylabel(y_label, fontsize=20)
            #ax_i.tick_params(labelsize=24)

            ax_i.tick_params(axis='both', which='major', labelsize=16)

            # Set x and y tick locators for better control
            from matplotlib.ticker import MaxNLocator
            ax_i.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Max 5 ticks on x-axis
            ax_i.yaxis.set_major_locator(MaxNLocator(nbins=3))  # Max 3 ticks on y-axis
            ax_i.grid(True, linestyle='--', alpha=0.6)  # Add a light grid for better readability

            ax_i.legend(fontsize=16)

            _save_legend = eval_df[hue].unique().shape[0] > 2
            if _save_legend:
                save_legend(ax_i, dir_name, 'legend.pdf')
                ax_i.legend_.remove()
            else:
                ax_i.legend(fontsize=5)

            if y_label == "MSE":
                ax_i.set_yscale('log')

            if y_label == "Relative Regret" and eval_df[y_label].max() > 10:
                ax_i.set_yscale('symlog')  # Symmetrical log scale with a threshold

                # Calculate the order of magnitude of max_y
                max_y = eval_df[y_label].max()
                min_y = eval_df[y_label].min()
                order_of_magnitude_max = 10 ** np.ceil(np.log10(max_y))  # Nearest larger power of 10
                order_of_magnitude_min = 10 ** np.ceil(np.log10(min_y)) if 10 ** np.ceil(np.log10(min_y)) >= 10 else 10
                # Calculate ticks using logarithmic interpolation
                num_ticks = 3  # Number of desired ticks
                if order_of_magnitude_min <= 1:
                    ticks = [0] + list(np.logspace(np.log10(order_of_magnitude_min), np.log10(order_of_magnitude_max), num=num_ticks - 1))
                else:
                    ticks = list(np.logspace(np.log10(order_of_magnitude_min), np.log10(order_of_magnitude_max), num=num_ticks))
                # Set ticks explicitly
                ax_i.set_yticks(ticks)

        fig.suptitle(title, fontsize=24)
        plt.tight_layout()
        save_and_show_plot(fig, dir_name, file_path, False)
    eval_df.to_csv(os.path.join(dir_name, file_path + '.csv'))


def save_and_show_plot(fig: Figure, folder: str, file_name: str, show: bool = False):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, file_name) + '.' + FIG_EXTENSION, bbox_inches='tight', format=FIG_EXTENSION)
    if show:
        plt.show()
    else:
        plt.close(fig=fig)
        fig.clf()



def time_plot(time_df: pd.DataFrame, folder: str, spacing_factor: float = 0.6):
    # Get unique dataset names, ensuring "OBD" is first
    real_datasets = list(time_df["Dataset"].unique())
    real_datasets.remove("OBD")
    dataset_order = ["OBD"] + real_datasets

    # Define x positions with spacing adjustment
    x_positions = np.arange(len(dataset_order), dtype=float)  # Base x positions
    x_positions[1:] += spacing_factor  # Shift non-"OBD" groups right

    # Get unique methods for color differentiation
    methods = time_df["Method"].unique()
    bar_width = 0.15  # Adjust bar width for multiple methods

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot bars manually for each method
    for i, method in enumerate(methods):
        subset = time_df[time_df["Method"] == method]
        y_values = [subset[subset["Dataset"] == ds]["Time [h]"].values[0] if ds in subset["Dataset"].values else 0 for
                    ds in dataset_order]

        ax.bar(x_positions + i * bar_width, y_values, width=bar_width, label=method, alpha=0.8)

    # Formatting
    ax.set_ylabel("Time [h]", fontsize=22)
    ax.set_xlabel("Datasets", fontsize=22)
    ax.tick_params(labelsize=20)
    ax.set_xticks(x_positions + (len(methods) - 1) * bar_width / 2)  # Center xticks
    ax.set_xticklabels(dataset_order, fontsize=20)
    ax.legend(fontsize=15)

    plt.tight_layout()
    save_and_show_plot(ax.get_figure(), folder, file_name="comp-time-exp" + str(time_df.shape[0]))



def plot_time_varying_log_data_size(log_dir_path, time_df):
    time_df.reset_index(inplace=True)
    ax = sns.lineplot(data=time_df, x='Logging Data Size', y='Time', hue='Method', linewidth=5, alpha=0.75, palette=PALETTE)
    ax.set_title('Time Comparison Varying Logging Data Size', fontsize=20)
    ax.set_xlabel('Logging Data Size', fontsize=20)
    ax.set_ylabel('Time [sec]', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=12)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_xlim(left=min(time_df['Logging Data Size']))
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35))
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    save_and_show_plot(ax.get_figure(), log_dir_path, 'time_varying_log_data_size')


def plot_performance_estimatorwise(estimator_selection_gt: dict, pi_e_info: dict, folder: str, file_name: str,
                                   title: str, x_label: str, y_label: str, y: str):
    alpha = 0.5
    all_policies = pd.DataFrame()
    for policy_key in estimator_selection_gt.keys():
        policy_df = estimator_selection_gt[policy_key].copy()
        policy_df[x_label] = pi_e_info[policy_key][1]
        all_policies = pd.concat([all_policies, policy_df]).reset_index(drop=True)

    estimators = list(all_policies["estimator_name"])
    hue_list = [name.split("_")[0].replace('Tuning', '') for name in estimators]
    style_list = ["_".join(name.split("_")[1:])
                  .replace('qmodel_', '')
                  .replace('RandomForestClassifier', 'RF')
                  .replace('LGBMClassifier', 'LGBM')
                  .replace('LogisticRegression', 'Logistic')
                  if len(name.split("_")) > 1 else "" for name in estimators]
    palette = dict(zip(sorted(np.unique(hue_list)), sns.color_palette(palette='Paired', n_colors=len(np.unique(hue_list)))))
    all_policies['style'] = style_list
    all_policies['hue'] = hue_list
    ax = sns.lineplot(x=x_label, y=y, hue='hue', ci=None, alpha=alpha, style='style',# dashes=True, markers=False,
                      data=all_policies, linewidth=2, palette=palette)  # default 95% C.I.
    ax.set_title(title, fontsize=25)
    plt.xticks(all_policies[x_label].unique())
    ax.set_xlabel(x_label, fontsize=25)
    ax.set_ylabel(y_label, fontsize=25)
    # if y_label == "Relative Regret":
    ax.set_yscale('log')
    ax.tick_params(labelsize=20)

    # get legend and change stuff
    handles, lables = ax.get_legend_handles_labels()
    for h in handles:
        h.set_alpha(alpha)
        h.set_linewidth(3)

    # replace legend using handles and labels from above
    plt.legend(handles, lables, borderaxespad=0, ncol=2, fontsize=10)#bbox_to_anchor=(1, 0), loc='lower right', )

    #plt.tight_layout()
    save_and_show_plot(ax.get_figure(), folder, file_name, False)