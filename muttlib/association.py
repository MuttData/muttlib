import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def cramers_corrected_stat(df, column1, column2, crosstab=True):
    """Returns cramers corrected v statistic association between two
    categorical variables from a dataframe.

    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe with the columns.
        This dataframe is expected to be a dataset with an observation in each
        row, in which case crosstab should be true.
        Or a previously crosstabbed dataframe.
    column1: str
        The first column's name.
    column2: str
        The second column's name.
    crosstab: bool (default true)
        If the dataframe has to be crosstabbed or not.
        If your dataframe is a dataset and you do not know what crosstab is,
        this has to be *true*.

    Returns
    -------
    float
        The statistic, a number between 0 and 1.
        Where 0 means no association and 1 means total association.
    """
    if crosstab:
        df = pd.crosstab(df[column1], df[column2])
    return cramers_corrected_stat_confusion(df)


def cramers_heatmap(df, columns, figsize=(12, 8), **kwargs):
    """Returns a heatmap showing cramers corrected v statistic association
    among categorical variables from a dataframe.

    Parameters
    ----------
    df: pandas.DataFrame
        A dataframe containing the columns.
        This dataframe is expected to be a dataset with an observation in each
        row.
    columns: [str]
        A list with the column names to compare.
    figsize: (int, int)
        The figsize to use in the plot.
    **kwargs: keyword arguments
        Arguments to forward to sns.heatmap.

    Returns
    -------
    matplotlib.pyplot.Axes
    """
    data = cramers_corrected_stat_heatmap(df, columns)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    sns.heatmap(data, annot=True, xticklabels=columns, yticklabels=columns,
                vmin=0, vmax=1, linewidths=0.5, ax=ax, **kwargs)
    return ax


def cramers_corrected_stat_confusion(confusion_matrix):
    """Calculate Cramers V statistic for categorical-categorical association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328

    https://stackoverflow.com/a/39266194
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return (phi2corr / min(kcorr - 1, rcorr - 1)) ** 0.5


def cramers_corrected_stat_heatmap(df, columns, crosstab=True):
    """Returns a heatmap showing cramers corrected v statistic association
    among categorical variables from a dataframe.

    Parameters
    ----------
    df: pandas.DataFrame
        A dataframe containing the columns.
        This dataframe is expected to be a dataset with an observation in each
        row.
    columns: [str]
        A list with the column names to compare.
    figsize: (int, int)
        The figsize to use in the plot.
    **kwargs: keyword arguments
        Arguments to forward to sns.heatmap.

    Returns
    -------
    [[float]]
        A matrix with the cramers values.
        The order is:
                 | column_1 | column_2 | ... | column_n
        column_1 |          |          | ... |
        column_2 |          |          | ... |
          ...    |          |          | ... |
        column_n |          |          | ... |
    """
    data = [[1 for _ in range(len(columns))] for _ in range(len(columns))]
    for i, column1 in enumerate(columns):
        for j, column2 in enumerate(columns[i + 1:], i + 1):
            stat = cramers_corrected_stat(df, column1, column2, crosstab)
            data[i][j] = data[j][i] = stat
    return data


def greedy_select(cramers_stats, max_association=0.9, response_ix=-1):
    """Returns the cramers_stats input filtering variables with high
    association with another variable.

    Parameters
    ----------
    cramers_stats: [[obj]]
        A list of lists like the output from *cramers_corrected_stat_heatmap*.
    max_association: float
        The maximum association value allowed in the returned list.
    response_ix: int
        The index in which the cramer stat can be found in the rows.

    Returns
    -------
    [[obj]]
        A list like cramers_stats list filtered in a new order.
    """
    result = []
    response_ix = response_ix if response_ix != -1 else len(cramers_stats) - 1
    stats = cramers_stats[response_ix]
    ordered = sorted(enumerate(stats), key=lambda x: x[1], reverse=True)
    ordered = [(ix, value) for ix, value in ordered if ix != response_ix]
    stats = stats[:response_ix] + stats[response_ix + 1:]
    blocked = set()
    while ordered:
        i, stat = ordered[0]
        ordered = ordered[1:]
        for j, _ in ordered:
            if cramers_stats[i][j] > max_association:
                blocked.add(j)
        ordered = [(j, c) for j, c in ordered if j not in blocked]
        result.append((i, stat))
    return result
