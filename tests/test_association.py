import random

from unittest import TestCase, main

import pandas as pd

from muttlib.association import cramers_corrected_stat_heatmap, greedy_select


class TestAssociation(TestCase):

    def test_cramers_v_matrix(self):
        """Taken from http://stats.lse.ac.uk/bergsma/pdf/cramerV3.pdf"""

        def diagonal_cell_proba(theta):
            return 1 / 9 + 2 * theta / 9

        def non_diagonal_cell_proba(theta):
            return 1 / 9 - theta / 9

        def random_theta(r):
            return random.uniform(-1 / (r - 1), 1)

        random.seed(14)

        # Cumulative probability table for a 3x3 matrix
        r = 3
        theta = random_theta(r)
        cumulative_probas = []
        cumulative = 0
        for i in range(3):
            for j in range(3):
                proba = (diagonal_cell_proba if i == j else
                         non_diagonal_cell_proba)
                cumulative += proba(theta)
                cumulative_probas.append(cumulative)

        # Fill the table with random values following the paper's probabilities
        n = 10000
        table = [[0 for _ in range(r)] for _ in range(r)]
        for _ in range(n):
            random_number = random.random()
            for ix, proba in enumerate(cumulative_probas):
                if random_number <= proba:
                    i, j = divmod(ix, r)
                    table[i][j] += 1
                    break
            else:
                table[-1][-1] += 1
        df = pd.DataFrame(table)

        # Calculate cramer's v
        cramers_matrix = cramers_corrected_stat_heatmap(df, df.columns, False)

        # Assert that diagonals are 1 and non diagonals close to abs(theta)
        expected_v = abs(theta)
        for i, row in enumerate(cramers_matrix):
            for j, item in enumerate(row):
                if i == j:
                    self.assertEqual(item, 1)
                else:
                    self.assertAlmostEqual(item, expected_v, places=2)

    def test_greedy_select_last_row(self):
        # The order would be: 0, 2, 1
        # 0 and 2 have an association of 0.7, let's filter over 0.6
        max_association = 0.6
        response_ix = 3
        cramers_matrix = [[1.0, 0.4, 0.7, 0.8],
                          [0.4, 1.0, 0.2, 0.4],
                          [0.7, 0.2, 1.0, 0.75],
                          [0.8, 0.4, 0.75, 1.0]]
        result = greedy_select(cramers_matrix, max_association, response_ix)
        expected = [(0, 0.8), (1, 0.4)]
        self.assertEqual(result, expected)

    def test_greedy_select_last_row_negative_index(self):
        # The order would be: 0, 2, 1
        # 0 and 2 have an association of 0.7, let's filter over 0.6
        # Then we would have: 0, 1
        max_association = 0.6
        response_ix = -1
        cramers_matrix = [[1.0, 0.4, 0.7, 0.8],
                          [0.4, 1.0, 0.2, 0.4],
                          [0.7, 0.2, 1.0, 0.75],
                          [0.8, 0.4, 0.75, 1.0]]
        result = greedy_select(cramers_matrix, max_association, response_ix)
        expected = [(0, 0.8), (1, 0.4)]
        self.assertEqual(result, expected)

    def test_greedy_select_middle_row(self):
        # The order would be: 0, 3, 2
        # 0 and 3 have an association of 0.7, let's filter over 0.6
        # Then we would have: 0, 2
        max_association = 0.6
        response_ix = 1
        cramers_matrix = [[1.0, 0.8, 0.3, 0.7],
                          [0.8, 1.0, 0.4, 0.75],
                          [0.3, 0.4, 1.0, 0.5],
                          [0.7, 0.75, 0.5, 1.0]]
        result = greedy_select(cramers_matrix, max_association, response_ix)
        expected = [(0, 0.8), (2, 0.4)]
        self.assertEqual(result, expected)

    def test_greedy_select_first_row(self):
        # The order would be: 1, 3, 2
        # 1 and 3 have an association of 0.7, let's filter over 0.6
        # Then we would have: 1, 2
        max_association = 0.6
        response_ix = 0
        cramers_matrix = [[1.0, 0.8, 0.4, 0.75],
                          [0.8, 1.0, 0.5, 0.7],
                          [0.4, 0.5, 1.0, 0.2],
                          [0.75, 0.7, 0.2, 1.0]]
        result = greedy_select(cramers_matrix, max_association, response_ix)
        expected = [(1, 0.8), (2, 0.4)]
        self.assertEqual(result, expected)


if __name__ == '__main__':
    main()
