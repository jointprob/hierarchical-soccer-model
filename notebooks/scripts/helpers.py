import numpy as np
import pandas as pd

from statsmodels.stats.proportion import proportion_confint
import calibration as cal

def format_season_df(df_orig):
    df = df_orig.copy()
    # Rename teams for consistency with original paper
    for col in ['Home', 'Away']:
        df[col] = df[col].map({'Hellas Verona': 'Verona', 'Internazionale': 'Inter'}).fillna(df[col])
    # Assign integer indices to alpha sorted team names
    teams_dict = {team_name: team_num for team_name, team_num in zip(sorted(df['Home'].unique().tolist()), range(df['Home'].nunique()))}
    for col in ['Home', 'Away']:
        df[col.lower()[0] + 'g'] = df[col].map(teams_dict)
    # Rename column names for consistency with original paper
    df['yg1'] = df['h_ftgoals'] ; df['yg2'] = df['a_ftgoals']
    # Sort the values to get an approximate match to the sorting order used in the paper
    df = df.sort_values(by=['Date', 'Home'], ascending=[True, False]).reset_index(drop=True)
    df['g'] = df.index
    # lowercase col names and select columns to go forward
    df.columns = [col.lower() for col in df.columns]
    df = df[['g', 'home', 'away', 'hg', 'ag', 'yg1', 'yg2']]
    # Put match outcome into df for later analysis
    condlist = [df['yg1']>df['yg2'], df['yg1']==df['yg2'], df['yg2']>df['yg1']]
    choicelist = ['hwin', 'draw', 'awin']
    df['result'] = np.select(condlist, choicelist)
    return df

def make_error_bars(bin_counts, error_bar_alpha, prob_pred, y_true):
    df = pd.DataFrame({'prob_pred': prob_pred, 'bin_count': bin_counts})
    ci_low, ci_upp = proportion_confint(df['prob_pred']*df['bin_count'], df['bin_count'], error_bar_alpha, method='normal')
    
    ci_low_num = round((error_bar_alpha / 2), 3) *100
    ci_upp_num = round(((1 - error_bar_alpha)*100 + ci_low_num), 3)
    df[str(ci_low_num) + '%_CI'] = ci_low
    df[str(ci_upp_num) + '%_CI'] = ci_upp
    
    return df

    
def make_calib_data(y_true, y_prob, n_bins, strategy, error_bar_alpha):
    # https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/calibration.py#L909
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    bin_counts = np.array([np.sum(binids==i) for i in np.unique(binids)])

    error_bars = make_error_bars(bin_counts, error_bar_alpha, prob_pred, y_true)

    df = pd.DataFrame({'mean_predicted_proba': prob_pred, 'mean_actual_proba': prob_true, 'bin_count': bin_counts})
    df = pd.concat([df, error_bars], axis=1)
    return df
    
    
def make_1x2_calib_data(df, n_bins=10, strategy='quantile',
                        error_bar_alpha=0.5,
                        pred_cols=['p(hwinc)', 'p(drawc)', 'p(awinc)'],
                        res_cols=['hwin', 'draw', 'awin']):
    cal_dfs = []
    for pred_col, res_col in zip(pred_cols, res_cols):
        cal_df = make_calib_data(df[res_col], df[pred_col], n_bins, strategy, error_bar_alpha).assign(calib=res_col)
        cal_df = cal_df.reset_index().rename(columns={'index': 'bin_num'})
        cal_df['calib'] = pd.Categorical(cal_df['calib'], categories=res_cols, ordered=True)
        cal_dfs.append(cal_df)
    df = pd.concat(cal_dfs, axis=0).reset_index(drop=True)
    return df


def get_1X2_CalMetrics(df, n_bins=20, pred_cols=['p(hwinc)', 'p(drawc)', 'p(awinc)'],
                       res_cols=['hwin', 'draw', 'awin']):
    col_dfs = []
    for pred_col, res_col in zip(pred_cols, res_cols):
        preds = df[pred_col] ; labels = df[res_col]
        ece = cal.get_ece(preds, labels, num_bins=n_bins)
        ece_em = cal.get_ece_em(preds, labels, num_bins=n_bins)
        cal_error = cal.get_calibration_error(preds, labels)
        [lower, median, upper] = cal.get_calibration_error_uncertainties(preds, labels, alpha=0.05)
        col_df = pd.DataFrame([[ece], [ece_em], [cal_error], [median, lower, upper]],
                              columns=['ece_metric', 'low_ci', 'upp_ci'],
                              index=['ece', 'ece_em', 'cal_error', 'median_ece']).assign(calib=pred_col)
        col_dfs.append(col_df)
    df = pd.concat(col_dfs, axis=0)
    return df