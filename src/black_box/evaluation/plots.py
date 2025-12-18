import os

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.optimize import curve_fit

from black_box.evaluation.metrics import metric_plot_label
from common.constants import FIG_EXTENSION
from common.evaluation.plots import save_and_show_plot


def correlation_plot(df: pd.DataFrame, plot_folder: str):
    plot = sns.heatmap(df.corr(numeric_only=True), annot=True, linewidths=.3, annot_kws={'fontsize': 'xx-small'})
    save_and_show_plot(plot.get_figure(), plot_folder, 'correlation', show=False)


def plot_feature_against_errors_grouped(x: pd.DataFrame, y: pd.DataFrame, plot_folder: str):
    df = x.join(y)
    cat_features = list(x.select_dtypes(include='object', exclude=np.number).columns)

    for feature in x.columns:
        for estimator in y.columns:
            if feature in cat_features:  # Categorical
                ax = sns.catplot(data=df, x=feature, y=estimator)
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            else:  # Numerical
                ax = sns.jointplot(data=df, x=feature, y=estimator, alpha=0.2)
                sns.move_legend(ax.ax_joint, "upper left", bbox_to_anchor=(1, 1))
        save_and_show_plot(ax.fig, plot_folder, feature + "", show=False)
        # plt.legend()


def plot_feature_against_errors_per_estimator(x: pd.DataFrame, y: pd.DataFrame, plot_folder: str, n_jobs):
    df = x.join(y)
    cat_features = list(x.select_dtypes(include='object', exclude=np.number).columns)
    df = preprocess(df, cat_features)
    plot_folder = plot_folder + '/single/'

    parallel = Parallel(n_jobs=n_jobs, verbose=True)
    parallel(delayed(parallelizable_plot_per_estimator)(cat_features, df, plot_folder, feature, y) for feature in
             x.columns)


def plot_feature_against_errors(x: pd.DataFrame, y: pd.DataFrame, plot_folder: str, n_jobs):
    df = x.join(y)
    cat_features = list(x.select_dtypes(include='object', exclude=np.number).columns)
    df = preprocess(df, cat_features)
    df = df.melt(id_vars=x.columns, value_vars=y.columns, value_name='error', var_name='estimator', ignore_index=False)
    df.reset_index(inplace=True, drop=True)
    cat_features = list(x.select_dtypes(include='object', exclude=np.number).columns)
    plot_folder = plot_folder + '/all/'

    parallel = Parallel(n_jobs=n_jobs, verbose=True)
    parallel(delayed(parallelizable_plot)(cat_features, df, plot_folder, feature) for feature in x.columns)


def preprocess(df: pd.DataFrame, cat_features):
    df_cat = df[cat_features]
    df = df.drop(cat_features, axis=1)
    df = df.dropna()
    columns = df.columns
    index = df.index
    df = df.to_numpy()
    df = np.add(df, np.zeros_like(df), out=1000000 * np.sign(df) * np.ones_like(df),
                where=(df < 1000000))
    df = pd.DataFrame(df, index=index, columns=columns)
    df = pd.concat([df, df_cat], axis=1)
    return df


def parallelizable_plot_per_estimator(cat_features, df, plot_folder, feature, y):
    plot_folder = plot_folder + '/' + feature + '/'
    for estimator in y.columns:
        if feature in cat_features:  # Categorical
            ax = sns.catplot(data=df, x=feature, y=estimator)
        else:  # Numerical
            ax = sns.jointplot(data=df, x=feature, y=estimator, alpha=0.2)
        save_and_show_plot(ax.fig, plot_folder, feature + "_" + str(estimator), show=False)


def parallelizable_plot(cat_features, df, plot_folder, feature):
    if feature in cat_features and feature not in 'error' and feature not in 'estimator':  # Categorical
        ax = sns.catplot(data=df, x=feature, y='error', hue='estimator')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    else:  # Numerical
        ax = sns.jointplot(data=df, x=feature, y='error', hue='estimator', alpha=0.2)
        sns.move_legend(ax.ax_joint, "upper left", bbox_to_anchor=(1.5, 1))
    save_and_show_plot(ax.fig, plot_folder, feature, show=False)


def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray, title: str, scores_str: str, model_folder: str,
                            file_prefix: str = ""):
    """Scatter plot of the predicted vs true targets."""
    max_y_true = np.max(y_true)
    max_y_pred = np.max(y_pred)
    min_y_true = np.min(y_true)
    min_y_pred = np.min(y_pred)
    margin_x = np.abs(max_y_true - min_y_true) * 0.1
    margin_y = np.abs(max_y_pred - min_y_pred) * 0.1
    fig, ax = plt.subplots()
    ax.plot([min_y_true, max_y_true], [min_y_true, max_y_true], "--r", linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.set_xlim(min_y_true - margin_x, max_y_true + margin_x)
    ax.set_ylim(min_y_pred - margin_y, max_y_pred + margin_y)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0)
    ax.legend([extra], [scores_str], loc="upper left")
    title = title + "\n Evaluation"
    ax.set_title(title)
    save_and_show_plot(fig, model_folder, file_prefix + "prediction", show=False)


def plot_classification_results(y_true: np.ndarray, y_pred: np.ndarray, model_folder: str, file_prefix: str = ""):
    """Scatter plot of the predicted vs true targets."""
    df = pd.DataFrame()
    df['True'] = y_true
    df['Predicted'] = y_pred
    ax = sns.stripplot(data=df, x='True', y='Predicted', order=np.unique(y_true).tolist(), alpha=0.2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    save_and_show_plot(ax.figure, model_folder, file_prefix + "prediction", show=False)


def plot_ranking_results(y_true: np.ndarray, y_pred: np.ndarray, model_folder: str, file_prefix: str = ""):
    """Scatter plot of the predicted vs true targets."""
    df = pd.DataFrame()
    df['True'] = y_true.ravel()
    df['Predicted'] = y_pred.ravel()
    ax = sns.scatterplot(data=df, x='True', y='Predicted', alpha=0.2)
    save_and_show_plot(ax.figure, model_folder, file_prefix + "prediction", show=False)


def evaluation_plots(results: pd.DataFrame, y: str, model_folder: str = "", file_prefix: str = ""):
    sns.set()
    ax = sns.lineplot(x='counterfactual beta', y=y, hue='method', data=results)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    save_and_show_plot(ax.figure, model_folder, file_prefix + y + "_plot", show=False)
    results.to_csv(os.path.join(model_folder, file_prefix + y + "_data.csv"))


def rand_jitter(arr):
    stdev = .02 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None,
           **kwargs):
    return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin,
                       vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)


def plot_train_size_varying_train_test_metrics(x, train_y, test_y, title, x_label, y_label, folder):
    linear_path = os.path.join(folder, 'linear')
    log_path = os.path.join(folder, 'log')
    os.makedirs(linear_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    df_train = pd.DataFrame({x_label: x, y_label: train_y})
    df_train['phase'] = 'Training'
    df_test = pd.DataFrame.from_dict({x_label: x, y_label: test_y})
    df_test['phase'] = 'Testing'
    df = pd.concat([df_train, df_test], axis=0)
    df.reset_index(inplace=True)
    ax = sns.lineplot(x=x_label, y=y_label, hue='phase', data=df, linewidth=5, alpha=0.75)
    ax.set_title(title)
    save_and_show_plot(ax.figure, linear_path, title.replace(" ", "-") + "_" + y_label)

    ax = sns.lineplot(x=x_label, y=y_label, hue='phase', data=df, linewidth=5, alpha=0.75)
    ax.set_title(title)
    ax.set(xscale='log')
    save_and_show_plot(ax.figure, log_path, title.replace(" ", "-") + "_" + y_label)


def plot_train_size_varying_experiment_res(res_gp_list: list[dict], res_train_list: list[DataFrame],
                                           res_test_list: list[DataFrame], train_size_list: list[int], metric: str,
                                           folder: str):
    x = train_size_list
    x_label = r"$|\mathcal{M}|$"
    train_y, test_y = [], []
    #for res_gp in res_gp_list:
    #    train_y.append(res_gp['train_fun'])
    #    test_y.append(res_gp['fun'])
    #plot_train_size_varying_train_test_metrics(x, train_y, test_y, "5-fold Cross-Validation (Bayesian Optimization)",
    #                                           x_label, metric_plot_label(metric), folder)

    for metric in res_train_list[0].columns:
        train_y, test_y = [], []
        for res_train, res_test in zip(res_train_list, res_test_list):
            train_y.append(res_train[metric][0])
            test_y.append(res_test[metric][0])
        plot_train_size_varying_train_test_metrics(x, train_y, test_y, 'Hold-out Training-Testing', x_label,
                                                   metric_plot_label(metric), folder)
        if metric == 'REGRET':
            exponential_fit_plot(x, test_y, None, x_label, 'Reg', folder)


def exponential_fit_plot(x, y, title, x_label, y_label, folder):
    palette = sns.color_palette()

    def exp_fun(t, a, b, c):
        return a * (t ** (-b)) + c
    x = np.array(x)
    y = np.array(y)
    lin = np.linspace(x.min(), x.max(), 200)
    popt, pcov = curve_fit(exp_fun, x, y, p0=(1, 0.1, 1), maxfev=1000000)

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, label='Original Samples', color=palette[0], linewidth=2, alpha=0.75)
    plt.plot(lin, exp_fun(lin, *popt), '--', label='Fit', c=palette[1], linewidth=4, alpha=0.75)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=15)
    from matplotlib.ticker import LogFormatterSciNotation
    plt.gca().yaxis.set_major_formatter(LogFormatterSciNotation())
    plt.gca().yaxis.set_minor_formatter(LogFormatterSciNotation())
    plt.gca().xaxis.set_major_formatter(LogFormatterSciNotation())
    plt.tick_params(axis='both', which='minor', labelsize=20)
    plt.tick_params(axis='both', labelsize=20)
    #plt.ticklabel_format(useOffset=False)
    plt.tight_layout()
    plt.grid(axis='y', which='minor')
    save_and_show_plot(plt.gcf(), folder, 'exp-fit-' + y_label)


def plot_feature_importance(importance, names, model_name, var, group, folder, plot_type, num_features):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    var = np.array(var)
    #tab10_colors = sns.color_palette("tab10")
    #custom_palette = [tab10_colors[0], tab10_colors[1], tab10_colors[3]]  # Blue, Red, Purple
    #sns.set_palette(custom_palette)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance, 'var': var, 'Features Groups': group}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    fi_df.reset_index(inplace=True, drop=True)
    fi_df = fi_df.iloc[0:num_features]

    # Define size of bar plot
    plt.figure(figsize=(8, 10))

    # Plot Searborn bar chart
    if plot_type == 'bar':
        ax = sns.barplot(data=fi_df, x='feature_importance', y='feature_names', alpha=0.75, hue='Features Groups', dodge=False, saturation=1)
                         #palette=sns.color_palette("mako", num_features))
        plt.legend(fontsize=15, loc="lower right")
        ax.errorbar(data=fi_df, x='feature_importance', y='feature_names', xerr='var', fmt="none", c="k",
                    linewidth=1, alpha=0.75)
    else:
        ax = None

    # Add chart labels
    #plt.title(model_name, fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=14)
    plt.xlabel('MDI', fontsize=24)
    plt.ylabel('Features', fontsize=24)
    ax.figure.tight_layout()
    save_and_show_plot(ax.figure, folder, file_name='feature_importance_' + plot_type + str(num_features))


def plot_feature_importance_up_down(importance, names, model_name, var, folder, plot_type, num_features, num_features_up_down):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    var = np.array(var)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance, 'var': var}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    fi_df.reset_index(inplace=True, drop=True)
    fi_df = fi_df.iloc[0:num_features]
    fi_df_up = fi_df.iloc[0:num_features_up_down]
    fi_df_down = fi_df.iloc[(fi_df.shape[0] - num_features_up_down):(fi_df.shape[0])]
    dots = pd.DataFrame.from_dict({'feature_names': ['...'], 'feature_importance': [0], 'var': [0]})

    fi_df = pd.concat([fi_df_up, dots, fi_df_down])

    # Define size of bar plot
    plt.figure(figsize=(10, 6))

    # Plot Searborn bar chart
    if plot_type == 'bar':
        ax = sns.barplot(data=fi_df, x='feature_importance', y='feature_names', alpha=0.85,
                         palette=sns.color_palette("mako", num_features_up_down * 2 + 1))
        ax.errorbar(data=fi_df, x='feature_importance', y='feature_names', xerr='var', fmt="none", c="k", linewidth=1, alpha=0.75)
    else:
        ax = None

    # Add chart labels
    plt.title(model_name + ' Feature Importance', fontsize=25)
    plt.xlabel('Feature Importance', fontsize=25)
    plt.ylabel('Features Names', fontsize=25)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=20)
    ax.figure.tight_layout()
    save_and_show_plot(ax.figure, folder, file_name='feature_importance_' + plot_type + '_up-down' +
                                                    str(num_features_up_down))



def plot_features_dist(plot_folder_path, x_train):
    x_train_visual = x_train.copy()

    # Drop all rows containing infinite values
    x_train_visual = x_train_visual[~x_train_visual.isin([np.inf, -np.inf]).any(axis=1)]

    # Drop all rows containing NaN values (optional)
    x_train_visual = x_train_visual.dropna()

    # Ensure layout values are integers for plotting
    cols = x_train_visual.shape[1]
    rows = int(np.ceil(cols / 4))

    # Create a larger figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))  # Increase figure size
    axes = axes.flatten()  # Flatten for easy indexing

    # Plot each feature
    for i, col in enumerate(x_train_visual.columns):
        x_train_visual[col].hist(ax=axes[i], bins=30, edgecolor='black')  # Histogram
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

    # Remove empty subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing

    # Save the figure
    os.makedirs(plot_folder_path, exist_ok=True)
    plt.savefig(os.path.join(plot_folder_path, "train_features_dist") + '.' + FIG_EXTENSION, bbox_inches='tight',
                format=FIG_EXTENSION)
