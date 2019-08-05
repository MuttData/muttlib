"""
Module to work, mangle, munge with DataFrames.

All functions should expect at least one arg to be a pd.DataFrame typed object.
"""
import logging
from pandas.core.groupby.generic import SeriesGroupBy

logger = logging.getLogger(f'df_utils.{__name__}')

SPECIAL_SERIES_AGGREGATIONS = {
    # Map name to aggr operation
    'count_nulls': lambda x: x.isnull().sum()
}
VALID_SERIES_AGGREGATION_NAMES = [
    a for a in SeriesGroupBy.__dict__.keys() if not a.startswith('_')
] + list(SPECIAL_SERIES_AGGREGATIONS)

# # Data Quality Checkers


def check_tolerated_data_symmetric_differences(
    df, df_expected, cols_to_tuples, max_tol
):
    """
    Check same values in df both dfs, no ordering.

    Returns check evaluation and message.
    Args:
        df (pd.DataFrame): the data.
        df_expected (pd.DataFrame): the expected data.
        cols_to_tuples (List[str]): columns list to evaluate.
        max_tol (float): max tolerance in the symmetric difference.

    Returns:
        (bool, str): evaluation and message output

    """
    expected_tuples_set = set(map(tuple, df_expected[cols_to_tuples].values.tolist()))
    response_tuples_set = set(map(tuple, df[cols_to_tuples].values.tolist()))
    sym_diff = expected_tuples_set.symmetric_difference(response_tuples_set)
    len_sym_diff = len(sym_diff)
    rv = len_sym_diff <= max_tol

    message = (
        f"There were {len_sym_diff} cases of differences, given the maximum "
        f"tolerance '{max_tol}' and the columns ({cols_to_tuples}) to evaluate."
    )
    return rv, message


def check_exact_expected_cols(df, expected_cols_l, with_ordering=False):
    """Check same columns in df than in list.

    Returns check evaluation and message.
    Args:
        df (pd.DataFrame): the data.
        expected_cols_l (List[str]): Columns list to expect.
        with_ordering (bool): optional, if column ordering matters

    Returns:
        (bool, str): evaluation and message output

    """
    df_cols = df.columns.tolist()
    if with_ordering is False:
        df_cols = set(df_cols)
        expected_cols_l = set(expected_cols_l)

    message = f"Input DF col list: {df_cols}.\n Expected: {expected_cols_l}."
    rv = df_cols == expected_cols_l
    return rv, message


def check_column_all_aggregated_within_range(
    df, grp_cols, agg_col, agg_op, min_tol, max_tol
):
    """
    If all aggregated values for each group are within a certain closed range.

    Returns check evaluation and message.
    Args:
        df (pd.DataFrame): the data.
        grp_cols (List[str]): columns list to group by.
        agg_col (str): column over which to aggregate.
        agg_op (str): pandas-valid aggregation operation.
        min_tol (float): min tolerance in the group's aggregated values.
        max_tol (float): max tolerance in the group's aggregated values.

    Returns:
        (bool, str): evaluation and message output

    """
    if agg_op not in VALID_SERIES_AGGREGATION_NAMES:
        raise ValueError(
            f" '{agg_op}' is not a valid pandas agg operation of the sort: "
            f"{VALID_SERIES_AGGREGATION_NAMES}"
        )
    if not isinstance(agg_col, str):
        raise ValueError(f"`agg_col` is not of string-type. Value passed was {agg_col}")

    # case of special aggregation operations, convert to appropriate lambda func
    if agg_op in SPECIAL_SERIES_AGGREGATIONS:
        agg_op = SPECIAL_SERIES_AGGREGATIONS[agg_op]

    agg_comparisons = (
        df.groupby(grp_cols)[agg_col].agg(agg_op).between(min_tol, max_tol)
    )

    rv = agg_comparisons.all()
    message = (
        f"There were {agg_comparisons.sum()} groups within tolerated ranges over "
        f"a total of {agg_comparisons.size} groups."
    )
    return rv, message
