"""Module to give FBProphet a common interface to Sklearn and general utilities
for forecasting problems like limiting the datasets to the last n days,
allowing wider grid search for hyperparameters not available using standard
FBProphet and Sklearn libraries.

Classes:
  - SkProphet: a wrapper around FBProphet to provide a scikit learn compatible
    API.
  - StepsSelectorEstimator: a scikit learn metaestimator to limit the amount of
    days used to fit a forecast. Wraps another estimator.

These two classes can be combined to perform gridsearch using FBProphet while
also exploring the amount of training days to use in the dataset.

The most relevant docstrings are on:
  - SkProphet.__init__
  - SkProphet.fit
  - StepsSelectorEstimator.__init__

Simple examples can be taken from the tests.
A complex example doing a grid search can be seen here:

.. code-block:: python

    import pandas as pd
    from sklearn.model_selection import GridSearchCV, ParameterGrid
    from muttlib.forecast import SkProphet, StepsSelectorEstimator

    # The grid has to be turned into a list if used in a StepsSelectorEstimator
    # as it has to be copyable for get / set params
    prophet_grid = list(ParameterGrid({
        'sk_date_column': ['date'],
        'sk_yhat_only': [True],
        'sk_extra_regressors': [
            [],
            [{'name': 'b'}],
            ],
        'prophet_kwargs': [
            dict(daily_seasonality='auto'),
            dict(daily_seasonality=True),
            ],
        }))

    days_selector_grid = {
        'estimator_class': [SkProphet],
        'amount_of_steps': [90, 120],
        'sort_col': ['date'],
        'estimator_kwargs': prophet_grid,
    }

    # To instance GridSearchCV, we need to pass an initialized estimator
    # (for example, a `StepsSelectorEstimator`)
    initial_estimator = StepsSelectorEstimator(
        SkProphet,
        days_selector_grid['amount_of_steps'][0],
        prophet_grid[0])
    cv = GridSearchCV(
        initial_estimator,
        days_selector_grid,
        cv=2,
        scoring='r2')

    X = pd.DataFrame({'date': [0, 2, 3, 4, 5], 'b': [1, 4, 5, 0, 9]})
    y = pd.Series([1, 1, 0, 1, 0])
    cv.fit(X, y)


TODO:
  - At the moment, given FBProphet's current version we have that the model's
    parameter for *extra_regressors* is not set on initialization but rather it
    is set by using a specific prophet method. Thus, we have that our current
    SKProphet class handles this parameter by setting it manually and knowing
    about this implicitly. If, for some future reason, prophet's API changes to
    include a variety of other/new parameters that are added _not-on-init _,
    then it'ld be probably a good idea to keep an internal dictionary of the
    parameter's dtype and prophet's method used to set it, so as to iterate and
    set these in a "programatic" way.
  - Evaluate if SkProphet.fit and SkProphet.copy default value should be False
    to save memory and cpu by default, risking to modifying the input data as a
    side effect of the function.
"""
from copy import deepcopy
from inspect import isclass, signature

import numpy as np
import pandas as pd

from fbprophet import Prophet
from sklearn.base import BaseEstimator


class SkProphet(Prophet):

    DS = 'ds'

    def __init__(
        self,
        sk_date_column=DS,
        sk_yhat_only=True,
        sk_extra_regressors=None,
        prophet_kwargs=None,
    ):
        """Scikit learn compatible interface for FBProphet.

        Parameters
        ----------
        sk_date_column: str
            Name of the column to use as date in Prophet.

        sk_yhat_only: Boolean
            True to return only the yhat from Prophet predictions.
            False to return everything.

        sk_extra_regressors: [] or [str] or [dict()]
            List with extra regressors to use. The list can have:

            * strings: column names (default prophet arguments for extra
              regressors will be used).
            * dicts: {name: *column_name*, prior_scale: _, standardize: _,
                      mode: _}

            For more information see Prophet.add_regressors.

        prophet_kwargs: dict
            Keyword arguments to forward to Prophet.
        """
        if sk_extra_regressors is None:
            sk_extra_regressors = []
        if prophet_kwargs is None:
            prophet_kwargs = {}

        super().__init__(**prophet_kwargs)
        self.sk_date_column = sk_date_column
        self.sk_yhat_only = sk_yhat_only
        self.sk_extra_regressors = sk_extra_regressors
        self.prophet_kwargs = prophet_kwargs
        self._set_my_extra_regressors()

    def fit(
        self, X, y=None, copy=True, **fit_params
    ):  # pylint: disable=arguments-differ
        """Scikit learn's like fit on the Prophet model.

        Parameters
        ----------
        X: pd.DataFrame
            A dataframe with the data to fit.
            It is expected to have a column with datetime values named as
            *self.sk_date_column*.
        y: None or str or (list, tuple, numpy.ndarray, pandas.Series/DataFrame)
           The label values to fit. If y is:
             - None: the column 'y' should be contained in X.
             - str: the name of the column to use in X.
             - list, tuple, ndarray, etc: the values to fit.
               If the values have two dimensions (a matrix instead of a vector)
               the first column will be used.
               E.g.: [1, 3] -> [1, 3] will be used.
               E.g.: [[1], [3]] -> [1, 3] will be used.
               E.g.: [[1, 2], [3, 4]] -> [1, 3] will be used.
        copy: Boolean
            True to copy the input dataframe before working with it to avoid
            modifying the original one.
            If True is set, X should contain the `ds` and `y` columns for
            prophet with those names.
            If False is provided, the input data will be copied and the copy
            modified if required.
        fit_params: keyword arguments
            Keyword arguments to forward to Prophet's fit.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Arg "X" passed can only be of pandas.DataFrame type.')
        if copy:
            X = X.copy()
        if self.sk_date_column != self.DS and self.sk_date_column in X.columns:
            X = X.rename({self.sk_date_column: self.DS}, axis=1)
        if y is not None:
            if isinstance(y, str) and y in X.columns:
                X = X.rename({y: 'y'}, axis=1)
            else:
                X['y'] = self._as_np_vector(y)
        return super().fit(X, **fit_params)

    def predict(self, X, copy=True):  # pylint: disable=arguments-differ
        """Scikit learn's predict (returns predicted values).

        Parameters
        ----------
        X: pandas.DataFrame
            Input data for predictions.
        copy: Boolean
            True to copy the input dataframe before working with it to avoid
            modifying the original one.
            If True is set, X should contain the `ds` and `y` columns for
            prophet with those names.
            If False is provided, the input data will be copied and the copy
            modified if required.
        """
        if copy:
            X = X.copy()
        if self.sk_date_column != self.DS and self.sk_date_column in X.columns:
            X = X.rename({self.sk_date_column: self.DS}, axis=1)
        predictions = super().predict(X)
        if self.sk_yhat_only:
            predictions = predictions.yhat.values
        return predictions

    def get_params(self, deep=True):
        """Scikit learn's get_params (returns the estimator's params)."""
        prophet_attrs = [
            attr for attr in signature(Prophet.__init__).parameters if attr != 'self'
        ]
        sk_attrs = [
            attr for attr in signature(self.__init__).parameters if attr != 'self'
        ]
        prophet_params = {a: getattr(self, a, None) for a in prophet_attrs}
        sk_params = {a: getattr(self, a, None) for a in sk_attrs}
        if deep:
            sk_params = deepcopy(sk_params)
            prophet_params = deepcopy(prophet_params)
        sk_params['prophet_kwargs'].update(prophet_params)
        return sk_params

    def set_params(self, **params):
        """Scikit learn's set_params (sets the parameters provided).
        Note on prophet keyword arguments precedence; this applies:
        - First, if some argument is explicitly provided, this value will be kept.
        - If not, but provided inside a 'prophet_kwargs' dict, the last is kept.
        - Lastly, if not provided in neither way but currently set, the value is not erased.
        """
        sk_kws = [
            attr for attr in signature(self.__init__).parameters if attr != 'self'
        ]
        current_prophet_kws = getattr(self, 'prophet_kwargs', {})
        explicit_prophet_kws = {}
        args_passed_prophet_kws = {}
        for attr, value in params.items():
            if attr == 'prophet_kwargs':
                explicit_prophet_kws = value
            elif attr not in sk_kws:
                args_passed_prophet_kws[attr] = value
            else:
                setattr(self, attr, value)
        prophet_kws = current_prophet_kws
        prophet_kws.update(explicit_prophet_kws)
        prophet_kws.update(args_passed_prophet_kws)
        for attr, value in prophet_kws.items():
            setattr(self, attr, value)
        setattr(self, 'prophet_kwargs', prophet_kws)
        self._set_my_extra_regressors()
        return self

    def _set_my_extra_regressors(self):
        """Adds the regressors defined in self.sk_extra_regressors.
        It is meant to be used at initialization.
        """
        if self.extra_regressors:
            self.extra_regressors = self.extra_regressors.__class__()
        for regressor in self.sk_extra_regressors:
            if isinstance(regressor, str):
                self.add_regressor(regressor)
            elif isinstance(regressor, dict):
                self.add_regressor(**regressor)
            else:
                raise TypeError(
                    'Invalid extra_regressor in SkProphet.'
                    'Extra regressors must be strings or dicts with '
                    '{name: *column_name*, prior_scale: _, standardize: _, '
                    'mode: _}'
                )

    def _as_np_vector(self, y):
        """Ensures a list, tuple, pandas.Series, pandas.DataFrame
        or numpy.ndarray is returned as a numpy.ndarray of dimension 1.

        Parameters
        ----------
        y: list, tuple, numpy.ndarray, pandas.Series, pandas.DataFrame
            The object containing the y values to fit.
            If y is multidimensional, e.g.: [[1, 2], [3, 4]], the first column
            will be returned as y value, continuining the example: [1, 3].

        Returns
        -------
        numpy.ndarray of dimension 1
            The values as a numpy array of dimension 1.
        """
        if isinstance(y, (list, tuple)):
            y = np.asarray(y)
        elif isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        if isinstance(y, np.ndarray):
            if len(y.shape) > 1:
                y = y[:, 0]
        return y

    def __repr__(self):
        """Text representation of the object to look it nicely in the
        interpreter.
        """
        return (
            f'{self.__class__.__name__}('
            f'sk_date_column="{self.sk_date_column}", '
            f'sk_yhat_only={self.sk_yhat_only}, '
            f'sk_extra_regressors={self.extra_regressors}'
            f'prophet_kwargs={self.prophet_kwargs})'
        )

    __str__ = __repr__


class StepsSelectorEstimator(BaseEstimator):
    def __init__(
        self, estimator_class, amount_of_steps, estimator_kwargs=None, sort_col='date'
    ):
        """An estimator that only uses a certain amount of rows on fit.

        Parameters
        ----------
        estimator_class: Classer or Estimator Class or estimator instance
            Estimator class to use to fit, if an Estimator Class is provided
            it will be wrapped with a metaestimator.Classer, if an instance
            is provided, its classed will be wrapped.
            examples:
            - Classer(sklearn.ensemble.RandomForestRegressor)
            - sklearn.ensemble.RandomForestRegressor
            - sklearn.ensemble.RandomForestRegressor()
        amount_of_steps: int
            The amount of time steps to use for training.
        sort_col: str
            Name of the column which will be used for sorting if X is a
            dataframe and has the column.
        estimator_kwargs: dict
            Keyword arguments to initialize EstimatorClass

        E.g.:

        > StepsSelectorEstimator(RandomForestRegressor(), 100)
        """
        if estimator_kwargs is None:
            estimator_kwargs = {}

        self.amount_of_steps = amount_of_steps
        self.sort_col = sort_col
        self.estimator_kwargs = estimator_kwargs
        self.estimator_class = Classer.from_obj(estimator_class)
        self._estimator = self.estimator_class.new(**self.estimator_kwargs)

    def fit(self, X, y):
        """Fits self.estimator only to the last self.amount_of_steps rows.
        Tries to sort X first.

        Parameters
        ----------
        X: pd.DataFrame
            A dataframe to fit.
        y: vector like
            Labels
        """
        if self.sort_col in X.columns:
            X = X.sort_values(self.sort_col, axis=0)
        index_to_drop = X.iloc[: -self.amount_of_steps].index
        y = y.drop(index_to_drop).reset_index(drop=True)
        X = X.drop(index_to_drop).reset_index(drop=True)
        self._estimator.fit(X, y)
        return self

    def predict(self, X):
        """Scikit's learn like predict."""
        return self._estimator.predict(X)

    def get_params(self, deep=True):
        """Get estimator params."""
        kwargs = self.estimator_kwargs
        if deep:
            kwargs = deepcopy(kwargs)
        return {
            'estimator_class': self.estimator_class,
            'amount_of_steps': self.amount_of_steps,
            'sort_col': self.sort_col,
            'estimator_kwargs': kwargs,
        }

    def set_params(self, **params):
        """Sets the estimator's params to \*\*params."""  # pylint: disable=anomalous-backslash-in-string
        self.estimator_class = Classer.from_obj(params['estimator_class'])
        self.amount_of_steps = params['amount_of_steps']
        self.sort_col = params['sort_col']
        self.estimator_kwargs = params['estimator_kwargs']
        self._estimator = self.estimator_class.new(**self.estimator_kwargs)
        return self

    def __repr__(self):  # pylint: disable=signature-differs
        """Text representation of the object to look it nicely in the
        interpreter.
        """
        return (
            f'{self.__class__.__name__}('
            f'estimator_class={Classer.from_obj(self.estimator_class)}, '
            f'amount_of_steps={self.amount_of_steps}, '
            f'estimator_kwargs={self.estimator_kwargs})'
        )

    __str__ = __repr__


class Classer:
    def __init__(self, EstimatorClass):
        """Wraps an EstimatorClass to avoid sklearn.base.clone exploting when
        called against an EstimatorClass during grid search of metaestimators.

        Parameters
        ----------
        EstimatorClass: class
            A Sklearn compatible estimator class.
        """
        self._class = EstimatorClass

    def new(self, *args, **kwargs):
        """Returns a new instance of the wrapped class initialized with the
        args and kwargs.
        """
        return self._class(*args, **kwargs)

    @classmethod
    def from_obj(cls, obj):
        """Initializes a new classer from an object, which can be another
        Classer, a class or an instance.
        """
        if isinstance(obj, Classer):
            return obj
        elif isclass(obj):
            return Classer(obj)
        else:
            return Classer(obj.__class__)

    def __eq__(self, other):
        """Equality checks inner class wrapped."""
        return self.__class__ == other.__class__ and self._class == other._class

    def __repr__(self):
        """Text representation of the object to look it nicely in the
        interpreter.
        """
        return f'{self.__class__.__name__}({self._class.__name__})'

    __str__ = __repr__
