import unittest

import numpy as np
import pandas as pd

from model_need import need_predictions


class NeedRankingTests(unittest.TestCase):
    def test_lower_income_wins_when_academics_and_effort_equal(self):
        candidates = pd.DataFrame({'id_candidat': [f'C{i:06}' for i in range(10)],
            'cote_r_equivalent': [30.] * 10, 'heures_travail_semaine': [10.] * 10,
            'revenu_familial_estime': [100000,90000,80000,70000,60000,50000,40000,30000,20000,10000]})
        selected = need_predictions(candidates)
        self.assertEqual(selected.sum(), 4)
        self.assertEqual(selected.tolist(), [0,0,0,0,0,0,1,1,1,1])

    def test_invalid_income_rejected(self):
        for bad in [0, -1, np.inf, np.nan]:
            candidates = pd.DataFrame({'id_candidat':['C000001'], 'cote_r_equivalent':[30.],
                'heures_travail_semaine':[10.], 'revenu_familial_estime':[bad]})
            with self.assertRaises(ValueError):
                need_predictions(candidates)


if __name__ == '__main__':
    unittest.main()
