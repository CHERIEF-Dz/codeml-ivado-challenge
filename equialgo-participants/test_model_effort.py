import unittest

import numpy as np
import pandas as pd

from model_academic import academic_predictions
from model_effort import effort_predictions


class EffortRankingTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'id_candidat': [f'C{i:06}' for i in range(10)],
            'cote_r_equivalent': [30,30,30,30,29,25,24,23,22,21],
            'heures_travail_semaine': [0,0,0,0,20,0,0,0,0,0]})

    def test_zero_weight_reproduces_academic_candidate(self):
        np.testing.assert_array_equal(effort_predictions(self.df, 0), academic_predictions(self.df))

    def test_effort_can_change_boundary_without_changing_budget(self):
        pred = effort_predictions(self.df, 0.144)
        self.assertEqual(pred.sum(), 4)
        self.assertEqual(pred[4], 1)
        self.assertEqual(pred[3], 0)

    def test_proxy_invariance_and_row_order_invariance(self):
        original = pd.Series(effort_predictions(self.df, 0.144), index=self.df.id_candidat)
        changed = self.df.assign(region_administrative='Cote-Nord', revenu_familial_estime=999999,
                                 code_postal_3='H1A', distance_domicile_campus_km=500)
        changed = changed.sample(frac=1, random_state=42)
        actual = pd.Series(effort_predictions(changed, 0.144), index=changed.id_candidat)
        pd.testing.assert_series_equal(original.sort_index(), actual.sort_index())

    def test_invalid_hours_and_coefficient_rejected(self):
        for value in [-1, np.nan, np.inf]:
            changed = self.df.astype({'heures_travail_semaine': float})
            changed.loc[0, 'heures_travail_semaine'] = value
            with self.assertRaises(ValueError):
                effort_predictions(changed, 0.144)
        with self.assertRaises(ValueError):
            effort_predictions(self.df, -0.1)


if __name__ == '__main__':
    unittest.main()
