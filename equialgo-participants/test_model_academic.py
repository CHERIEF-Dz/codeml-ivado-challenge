import unittest

import numpy as np
import pandas as pd

from model_academic import academic_predictions


class AcademicRankingTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'id_candidat': [f'C{i:06}' for i in range(10)],
                                'cote_r_equivalent': [30,30,30,30,30,25,24,23,22,21]})

    def test_budget_and_ties_are_independent_of_input_order(self):
        first = pd.Series(academic_predictions(self.df), index=self.df.id_candidat)
        shuffled = self.df.sample(frac=1, random_state=42)
        second = pd.Series(academic_predictions(shuffled), index=shuffled.id_candidat)
        pd.testing.assert_series_equal(first.sort_index(), second.sort_index())
        self.assertEqual(first.sum(), 4)
        self.assertEqual(first.iloc[:4].tolist(), [1,1,1,1])

    def test_proxy_changes_have_no_effect(self):
        changed = self.df.assign(region_administrative='Montreal', revenu_familial_estime=500000,
                                 heures_travail_semaine=40, distance_domicile_campus_km=300)
        np.testing.assert_array_equal(academic_predictions(self.df), academic_predictions(changed))

    def test_duplicate_ids_and_missing_scores_fail(self):
        duplicate = self.df.copy()
        duplicate.loc[1, 'id_candidat'] = duplicate.loc[0, 'id_candidat']
        with self.assertRaises(ValueError):
            academic_predictions(duplicate)
        missing = self.df.copy()
        missing.loc[0, 'cote_r_equivalent'] = np.nan
        with self.assertRaises(ValueError):
            academic_predictions(missing)


if __name__ == '__main__':
    unittest.main()
