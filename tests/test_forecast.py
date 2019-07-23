from datetime import datetime, timedelta
from unittest import TestCase, main

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from muttlib.forecast import SkProphet, Classer, DaysSelectorEstimator


class TestSkProphet(TestCase):

    def test_init(self):
        sk_date_column = 'ds'
        sk_yhat_only = True
        sk_extra_regressors = []
        prophet_kwargs = {'daily_seasonality': True}
        p = SkProphet(sk_date_column, sk_yhat_only, sk_extra_regressors,
                      prophet_kwargs)
        # No extra regressors
        self.assertTrue(p.daily_seasonality)
        self.assertFalse(p.extra_regressors)

    def test_init_extra_regressors_str(self):
        sk_date_column = 'ds'
        sk_yhat_only = True
        sk_extra_regressors = ['x']
        prophet_kwargs = {'daily_seasonality': True}
        p = SkProphet(sk_date_column, sk_yhat_only, sk_extra_regressors,
                      prophet_kwargs)
        # Extra regressors are set on initialization
        self.assertTrue(p.extra_regressors)

    def test_init_extra_regressors_dict(self):
        sk_date_column = 'ds'
        sk_yhat_only = True
        sk_extra_regressors = [dict(name='x', mode='multiplicative')]
        prophet_kwargs = {'daily_seasonality': True}
        p = SkProphet(sk_date_column, sk_yhat_only, sk_extra_regressors,
                      prophet_kwargs)
        # Extra regressors are set on initialization
        self.assertTrue(p.extra_regressors)

    def test_fit_X_only(self):
        p = SkProphet('ds', True, [], {})
        X = self._get_dataset()
        p2 = p.fit(X)
        self.assertEqual(p, p2)

    def test_fit_X_only_rename_ds(self):
        p = SkProphet('date', True, [], {})
        X = self._get_dataset()
        X = X.rename({'ds': 'date'}, axis=1)
        p2 = p.fit(X)
        self.assertEqual(p, p2)

    def test_fit_y_str(self):
        p = SkProphet('date', True, [], {})
        X = self._get_dataset()
        X = X.rename({'y': 'mica'}, axis=1)
        p2 = p.fit(X, 'mica')
        self.assertEqual(p, p2)

    def test_fit_y_list(self):
        p = SkProphet('date', True, [], {})
        X = self._get_dataset()
        p2 = p.fit(X[['ds', 'x']], X.y.values.tolist())
        self.assertEqual(p, p2)

    def test_estimate(self):
        p = SkProphet('ds', True, [], {})
        X = self._get_dataset()
        y_pred = p.fit(X).predict(X)
        self.assertIsInstance(y_pred, np.ndarray)

    def test_estimate_full_output(self):
        p = SkProphet('ds', False, [], {})
        X = self._get_dataset()
        y_pred = p.fit(X).predict(X)
        self.assertIsInstance(y_pred, pd.DataFrame)

    def test_estimate_rename_ds(self):
        p = SkProphet('date', True, [], {})
        X = self._get_dataset()
        X = X.rename({'ds': 'date'}, axis=1)
        y_pred = p.fit(X).predict(X)
        self.assertIsInstance(y_pred, np.ndarray)

    def test_as_np_vector_list_1d(self):
        p = SkProphet('date', True, [], {})
        y = [1, 2, 3]
        _y = p._as_np_vector(y)
        self.assertIsInstance(_y, np.ndarray)
        self.assertEqual(len(_y.shape), 1)
        self.assertEqual(_y.tolist(), [1, 2, 3])

    def test_as_np_vector_series(self):
        p = SkProphet('date', True, [], {})
        y = pd.Series([1, 2, 3])
        _y = p._as_np_vector(y)
        self.assertIsInstance(_y, np.ndarray)
        self.assertEqual(len(_y.shape), 1)
        self.assertEqual(_y.tolist(), [1, 2, 3])

    def test_as_np_vector_dataframe_1d(self):
        p = SkProphet('date', True, [], {})
        y = pd.DataFrame({'a': [1, 2, 3]})
        _y = p._as_np_vector(y)
        self.assertIsInstance(_y, np.ndarray)
        self.assertEqual(len(_y.shape), 1)
        self.assertEqual(_y.tolist(), [1, 2, 3])

    def test_as_np_vector_list_2d(self):
        p = SkProphet('date', True, [], {})
        y = [[1, 4], [2, 5], [3, 6]]
        _y = p._as_np_vector(y)
        self.assertIsInstance(_y, np.ndarray)
        self.assertEqual(len(_y.shape), 1)
        self.assertEqual(_y.tolist(), [1, 2, 3])

    def test_as_np_vector_dataframe_2d(self):
        p = SkProphet('date', True, [], {})
        y = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        _y = p._as_np_vector(y)
        self.assertIsInstance(_y, np.ndarray)
        self.assertEqual(len(_y.shape), 1)
        self.assertEqual(_y.tolist(), [1, 2, 3])

    def test_as_np_vector_ndarray_1d(self):
        p = SkProphet('date', True, [], {})
        y = np.array([1, 2, 3])
        _y = p._as_np_vector(y)
        self.assertIsInstance(_y, np.ndarray)
        self.assertEqual(len(_y.shape), 1)
        self.assertEqual(_y.tolist(), [1, 2, 3])

    def test_as_np_vector_ndarray_2d(self):
        p = SkProphet('date', True, [], {})
        y = np.array([[1, 4], [2, 5], [3, 6]])
        _y = p._as_np_vector(y)
        self.assertIsInstance(_y, np.ndarray)
        self.assertEqual(len(_y.shape), 1)
        self.assertEqual(_y.tolist(), [1, 2, 3])

    def test_get_set_params(self):
        sk_date_column = 'ds'
        sk_yhat_only = True
        sk_extra_regressors = [dict(name='x', mode='multiplicative')]
        prophet_kwargs = {'daily_seasonality': True}
        p = SkProphet(sk_date_column, sk_yhat_only, sk_extra_regressors,
                      prophet_kwargs)
        # Assert deep
        params = p.get_params()
        self.assertEqual(len(params), 4)
        self.assertEqual(params['sk_extra_regressors'], sk_extra_regressors)
        self.assertEqual(params['prophet_kwargs'], prophet_kwargs)

    @staticmethod
    def _get_dataset():
        n = 60
        X = pd.DataFrame({
            'ds': [datetime(2019, 1, 1) + timedelta(days=i) for i in range(n)],
            'x': list(range(n)),
            'y': [i + x_i ** 2 for i, x_i in enumerate(range(n))]})
        return X


class TestDaysSelectorEstimator(TestCase):

    def test_init(self):
        # Arguments
        estimator_class = LinearRegression
        amount_of_days = 30
        estimator_kwargs = {'fit_intercept': False}
        sort_col = 'date'
        # Initialize
        ds = DaysSelectorEstimator(
            estimator_class, amount_of_days, estimator_kwargs, sort_col)
        # Asserts
        self.assertEqual(ds._estimator.__class__, estimator_class)
        self.assertFalse(ds._estimator.fit_intercept)
        self.assertEqual(ds.sort_col, sort_col)
        self.assertEqual(ds.estimator_kwargs, estimator_kwargs)
        self.assertTrue(ds.estimator_kwargs is estimator_kwargs)

    def test_estimate(self):
        # Fake data with a v shape. First 30 drop. Last 30 Increase.
        x = list(range(30)) + list(range(30))
        y = pd.Series(list(range(29, -1, -1)) + list(range(30)))
        X = pd.DataFrame({'x': x})
        # Initialize
        estimator_class = LinearRegression
        ds = DaysSelectorEstimator(estimator_class, 30)
        # Fit
        ds.fit(X, y)
        # Predict
        y_pred = ds.predict(X)
        # Assert that only the increasing slope was fitted
        for i in range(60):
            self.assertAlmostEqual(y_pred[i], i % 30)

    def test_get_set_params(self):
        # Initialize
        estimator_class = LinearRegression
        amount_of_days = 30
        estimator_kwargs = {'fit_intercept': False}
        sort_col = 'date'
        ds = DaysSelectorEstimator(
            estimator_class, amount_of_days, estimator_kwargs, sort_col)
        # Assert deep
        params = ds.get_params()
        self.assertEqual(len(params), 4)
        self.assertEqual(params['estimator_class'], Classer(estimator_class))
        self.assertEqual(params['amount_of_days'], amount_of_days)
        self.assertEqual(params['estimator_kwargs'], estimator_kwargs)
        self.assertTrue(params['estimator_kwargs'] is not estimator_kwargs)
        self.assertEqual(params['sort_col'], sort_col)
        # Assert shallow
        params = ds.get_params(deep=False)
        self.assertTrue(params['estimator_kwargs'] is estimator_kwargs)


if __name__ == '__main__':
    main()
