# TODO: migrate to pytest
from datetime import datetime, timedelta
from unittest import TestCase, main
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression

from muttlib.forecast import SkProphet, Classer, StepsSelectorEstimator


class TestSkProphet(TestCase):
    def test_init(self):
        sk_date_column = 'ds'
        sk_yhat_only = True
        sk_extra_regressors = []
        prophet_kwargs = {'daily_seasonality': True}
        p = SkProphet(sk_date_column, sk_yhat_only, sk_extra_regressors, prophet_kwargs)
        # No extra regressors
        self.assertTrue(p.daily_seasonality)
        self.assertFalse(p.extra_regressors)

    def test_init_extra_regressors_str(self):
        sk_date_column = 'ds'
        sk_yhat_only = True
        sk_extra_regressors = ['x']
        prophet_kwargs = {'daily_seasonality': True}
        p = SkProphet(sk_date_column, sk_yhat_only, sk_extra_regressors, prophet_kwargs)
        # Extra regressors are set on initialization
        self.assertTrue(p.extra_regressors)

    def test_init_extra_regressors_dict(self):
        sk_date_column = 'ds'
        sk_yhat_only = True
        sk_extra_regressors = [dict(name='x', mode='multiplicative')]
        prophet_kwargs = {'daily_seasonality': True}
        p = SkProphet(sk_date_column, sk_yhat_only, sk_extra_regressors, prophet_kwargs)
        # Extra regressors are set on initialization
        self.assertTrue(p.extra_regressors)

    def test_init_extra_regressors_type_error(self):
        sk_date_column = 'ds'
        sk_yhat_only = True
        sk_extra_regressors = [4]
        prophet_kwargs = {'daily_seasonality': True}
        # Extra regressors of a bad type
        with self.assertRaises(TypeError):
            SkProphet(sk_date_column, sk_yhat_only, sk_extra_regressors, prophet_kwargs)

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

    def test_fit_X_not_dataframe_error(self):
        p = SkProphet('ds', True, [], {})
        X = [[1, 2], [3, 4]]
        with self.assertRaises(TypeError):
            p.fit(X)

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

    def test_get_params_deep(self):
        sk_date_column = 'ds'
        sk_yhat_only = True
        sk_extra_regressors = [dict(name='x', mode='multiplicative')]
        prophet_kwargs = {'daily_seasonality': True}
        p = SkProphet(sk_date_column, sk_yhat_only, sk_extra_regressors, prophet_kwargs)
        # Assert deep equality
        params = p.get_params()
        self.assertEqual(len(params), 4)
        self.assertEqual(params['sk_date_column'], sk_date_column)
        self.assertEqual(params['sk_yhat_only'], sk_yhat_only)
        self.assertEqual(params['sk_extra_regressors'], sk_extra_regressors)
        self.assertEqual(
            params['prophet_kwargs']['daily_seasonality'],
            prophet_kwargs['daily_seasonality'],
        )
        # Assert deep different objects
        self.assertTrue(params['sk_extra_regressors'] is not sk_extra_regressors)
        self.assertTrue(params['sk_extra_regressors'][0] is not sk_extra_regressors[0])
        self.assertTrue(params['prophet_kwargs'] is not prophet_kwargs)

    def test_get_params_shallow(self):
        sk_date_column = 'ds'
        sk_yhat_only = True
        sk_extra_regressors = [dict(name='x', mode='multiplicative')]
        prophet_kwargs = {'daily_seasonality': True}
        p = SkProphet(sk_date_column, sk_yhat_only, sk_extra_regressors, prophet_kwargs)
        # Assert shallow equality
        params = p.get_params(deep=False)
        self.assertEqual(len(params), 4)
        self.assertEqual(params['sk_date_column'], sk_date_column)
        self.assertEqual(params['sk_yhat_only'], sk_yhat_only)
        self.assertEqual(params['sk_extra_regressors'], sk_extra_regressors)
        self.assertEqual(
            params['prophet_kwargs']['daily_seasonality'],
            prophet_kwargs['daily_seasonality'],
        )
        # Assert shallow same objects
        self.assertTrue(params['sk_date_column'] is sk_date_column)
        self.assertTrue(params['sk_extra_regressors'] is sk_extra_regressors)
        self.assertTrue(params['sk_extra_regressors'][0] is sk_extra_regressors[0])
        self.assertTrue(params['prophet_kwargs'] is prophet_kwargs)

    def test_set_params(self):
        sk_date_column = 'ds'
        sk_yhat_only = True
        sk_extra_regressors = [dict(name='x', mode='multiplicative')]
        prophet_kwargs = {'daily_seasonality': True}
        p = SkProphet(sk_date_column, sk_yhat_only, sk_extra_regressors, prophet_kwargs)
        # Alternative params
        alternative_params = {
            'sk_date_column': 'date',
            'sk_extra_regressors': [dict(name='z')],
        }
        new_params = p.get_params(deep=True)
        new_params.update(alternative_params)
        # Assert params
        p.set_params(**new_params)
        params = p.get_params(deep=False)
        self.assertEqual(new_params, params)
        self.assertEqual(p.sk_date_column, 'date')
        self.assertEqual(len(p.sk_extra_regressors), 1)
        self.assertEqual(p.sk_extra_regressors[0]['name'], 'z')

    def test_prophet_kwargs_set_params(self):
        prophet_original_pars = {'n_changepoints': 10}
        m = SkProphet(prophet_kwargs=prophet_original_pars)
        prophet_new_pars = {
            'prophet_kwargs': {'yearly_seasonality': 4},
            'n_changepoints': 2,
        }
        m.set_params(**prophet_new_pars)
        expected_prophet_kwargs = {'yearly_seasonality': 4, 'n_changepoints': 2}
        self.assertEqual(m.prophet_kwargs, expected_prophet_kwargs)
        self.assertEqual(m.yearly_seasonality, 4)
        self.assertEqual(m.n_changepoints, 2)

    def test_repr(self):
        # Initialize
        sk_date_column = 'date'
        sk_yhat_only = False
        sk_extra_regressors = [dict(name='x', mode='multiplicative')]
        prophet_kwargs = {'daily_seasonality': True}
        skprophet = SkProphet(
            sk_date_column, sk_yhat_only, sk_extra_regressors, prophet_kwargs
        )
        # Assert: do not assert prophet internal attributes
        expected = (
            'SkProphet('
            'sk_date_column="date", '
            'sk_yhat_only=False, '
            'sk_extra_regressors='
        )
        self.assertEqual(str(skprophet)[: len(expected)], expected)

    @staticmethod
    def _get_dataset():
        n = 60
        X = pd.DataFrame(
            {
                'ds': [datetime(2019, 1, 1) + timedelta(days=i) for i in range(n)],
                'x': list(range(n)),
                'y': [i + x_i ** 2 for i, x_i in enumerate(range(n))],
            }
        )
        return X


class TestStepsSelectorEstimator(TestCase):
    def test_init(self):
        # Arguments
        estimator_class = LinearRegression
        amount_of_steps = 30
        estimator_kwargs = {'fit_intercept': False}
        sort_col = 'date'
        # Initialize
        se = StepsSelectorEstimator(
            estimator_class, amount_of_steps, estimator_kwargs, sort_col
        )
        # Asserts
        self.assertEqual(se._estimator.__class__, estimator_class)
        self.assertFalse(se._estimator.fit_intercept)
        self.assertEqual(se.sort_col, sort_col)
        self.assertEqual(se.estimator_kwargs, estimator_kwargs)
        self.assertTrue(se.estimator_kwargs is estimator_kwargs)

    def test_fit_unsorted(self):
        # Fake data with a v shape. First 30 drop. Last 30 Increase.
        x = list(range(30)) + list(range(30))
        y = pd.Series(list(range(29, -1, -1)) + list(range(30)))
        X = pd.DataFrame({'x': x})
        # Initialize
        estimator_class = LinearRegression
        steps_estimator = StepsSelectorEstimator(estimator_class, 30)
        steps_estimator._estimator = MagicMock()
        # Fit
        steps_estimator.fit(X, y)
        # Assert
        args, kwargs = steps_estimator._estimator.fit.call_args
        self.assertEqual(len(args), 2)
        self.assertEqual(len(kwargs), 0)
        X_call, y_call = args
        self.assertEqual(X.iloc[-30:].values.tolist(), X_call.values.tolist())
        self.assertEqual(y.iloc[-30:].values.tolist(), y_call.values.tolist())

    def test_fit_sorted(self):
        # Fake data with a v shape. First 30 drop. Last 30 Increase.
        x = list(range(30)) + list(range(30))
        y = pd.Series(list(range(29, -1, -1)) + list(range(30)))
        X = pd.DataFrame({'x': x})
        # Initialize
        estimator_class = LinearRegression
        steps_estimator = StepsSelectorEstimator(estimator_class, 30, sort_col='x')
        steps_estimator._estimator = MagicMock()
        # Fit
        steps_estimator.fit(X, y)
        # Assert
        args, kwargs = steps_estimator._estimator.fit.call_args
        self.assertEqual(len(args), 2)
        self.assertEqual(len(kwargs), 0)
        X_call, y_call = args
        self.assertEqual(
            [[i] for i in range(15, 30) for _ in range(2)], X_call.values.tolist()
        )
        self.assertEqual(
            list(range(14, -1, -1)) + list(range(15, 30, 1)), y_call.values.tolist()
        )

    def test_estimate(self):
        # Fake data with a v shape. First 30 drop. Last 30 Increase.
        x = list(range(30)) + list(range(30))
        y = pd.Series(list(range(29, -1, -1)) + list(range(30)))
        X = pd.DataFrame({'x': x})
        # Initialize
        estimator_class = LinearRegression
        steps_estimator = StepsSelectorEstimator(estimator_class, 30)
        # Fit
        steps_estimator.fit(X, y)
        # Predict
        y_pred = steps_estimator.predict(X)
        # Assert that only the increasing slope was fitted
        for i in range(60):
            self.assertAlmostEqual(y_pred[i], i % 30)

    def test_get_params_deep(self):
        # Initialize
        estimator_class = LinearRegression
        amount_of_steps = 30
        estimator_kwargs = {'fit_intercept': False}
        sort_col = 'date'
        steps_estimator = StepsSelectorEstimator(
            estimator_class, amount_of_steps, estimator_kwargs, sort_col
        )
        # Assert deep
        params = steps_estimator.get_params()
        self.assertEqual(len(params), 4)
        self.assertEqual(params['estimator_class'], Classer(estimator_class))
        self.assertEqual(params['amount_of_steps'], amount_of_steps)
        self.assertEqual(params['estimator_kwargs'], estimator_kwargs)
        self.assertTrue(params['estimator_kwargs'] is not estimator_kwargs)
        self.assertEqual(params['sort_col'], sort_col)

    def test_get_params_shallow(self):
        # Initialize
        estimator_class = LinearRegression
        amount_of_steps = 30
        estimator_kwargs = {'fit_intercept': False}
        sort_col = 'date'
        steps_estimator = StepsSelectorEstimator(
            estimator_class, amount_of_steps, estimator_kwargs, sort_col
        )
        # Assert shallow
        params = steps_estimator.get_params(deep=False)
        self.assertEqual(params['estimator_class'], Classer(estimator_class))
        self.assertEqual(params['amount_of_steps'], amount_of_steps)
        self.assertEqual(params['estimator_kwargs'], estimator_kwargs)
        self.assertTrue(params['estimator_kwargs'] is estimator_kwargs)
        self.assertEqual(params['sort_col'], sort_col)

    def test_set_params(self):
        # Initialize
        estimator_class = LinearRegression
        amount_of_steps = 30
        estimator_kwargs = {'fit_intercept': False}
        sort_col = 'date'
        steps_estimator = StepsSelectorEstimator(
            estimator_class, amount_of_steps, estimator_kwargs, sort_col
        )
        # Set params
        estimator_class = LogisticRegression
        amount_of_steps = 40
        estimator_kwargs = {'fit_intercept': True}
        sort_col = 'month'
        steps_estimator.set_params(
            estimator_class=LogisticRegression,
            amount_of_steps=amount_of_steps,
            estimator_kwargs=estimator_kwargs,
            sort_col=sort_col,
        )
        # Assert deep
        params = steps_estimator.get_params()
        self.assertEqual(len(params), 4)
        self.assertEqual(params['estimator_class'], Classer(estimator_class))
        self.assertEqual(params['amount_of_steps'], amount_of_steps)
        self.assertEqual(params['estimator_kwargs'], estimator_kwargs)
        self.assertEqual(params['sort_col'], sort_col)

    def test_repr(self):
        # Initialize
        estimator_class = LinearRegression
        amount_of_steps = 30
        estimator_kwargs = {'fit_intercept': False}
        sort_col = 'date'
        steps_estimator = StepsSelectorEstimator(
            estimator_class, amount_of_steps, estimator_kwargs, sort_col
        )
        # Assert
        expected = (
            'StepsSelectorEstimator('
            'estimator_class=Classer(LinearRegression), '
            'amount_of_steps=30, '
            'estimator_kwargs={\'fit_intercept\': False})'
        )
        self.assertEqual(str(steps_estimator), expected)


class TestClasser(TestCase):
    def test_init(self):
        c = Classer(LinearRegression)
        self.assertEqual(c._class, LinearRegression)

    def test_new(self):
        c = Classer(LinearRegression)
        estimator = c.new(fit_intercept=False)
        self.assertEqual(estimator.__class__, LinearRegression)
        self.assertEqual(estimator.fit_intercept, False)

    def test_equality(self):
        classer_a = Classer(LinearRegression)
        classer_b = Classer(LinearRegression)
        self.assertEqual(classer_a, classer_b)

    def test_inequality_classers(self):
        classer_a = Classer(LinearRegression)
        classer_b = Classer(LogisticRegression)
        self.assertNotEqual(classer_a, classer_b)

    def test_inequality_others(self):
        classer_a = Classer(LinearRegression)
        self.assertNotEqual(classer_a, 4)

    def test_from_obj_classer(self):
        classer_a = Classer(LinearRegression)
        classer_b = Classer.from_obj(classer_a)
        self.assertEqual(classer_a, classer_b)

    def test_from_obj_class(self):
        classer_a = Classer(LinearRegression)
        classer_b = Classer.from_obj(LinearRegression)
        self.assertEqual(classer_a, classer_b)

    def test_from_obj_instancec(self):
        classer_a = Classer(LinearRegression)
        classer_b = Classer.from_obj(LinearRegression())
        self.assertEqual(classer_a, classer_b)

    def test_repr(self):
        classer_a = Classer(LinearRegression)
        self.assertEqual(str(classer_a), 'Classer(LinearRegression)')


if __name__ == '__main__':
    main()
