import unittest

import numpy as np
import pandas as pd

from model_improved import allocation, features, metrics, models, ROOT, NUM, CAT
from model_merit import neutral_scores


class AllocationTests(unittest.TestCase):
    def test_budget_boundaries_on_submission_size(self):
        scores = np.linspace(0, 1, 4000)
        groups = np.zeros(4000, dtype=bool)
        for rate, count in [(0.36, 1440), (0.40, 1600), (0.44, 1760)]:
            result = allocation(scores, groups, rate)
            self.assertEqual(result.sum(), count)
            self.assertTrue(np.all(result[-count:] == 1))

    def test_ties_are_deterministic(self):
        self.assertEqual(allocation(np.ones(10), np.zeros(10)).tolist(), [1]*4 + [0]*6)

    def test_regional_adjustment_preserves_budget(self):
        scores = np.linspace(0, 1, 10)
        remote = np.arange(10) < 4
        pred = allocation(scores, remote, remote_bonus=2)
        self.assertEqual(pred.sum(), 4)
        self.assertTrue(np.all(pred[remote] == 1))

    def test_invalid_scores_and_budget_rejected(self):
        with self.assertRaises(ValueError):
            allocation([0, np.nan], [False, True])
        with self.assertRaises(ValueError):
            allocation([0, 1], [False, True], rate=0.5)

    def test_metrics_against_hand_calculation(self):
        result = metrics([1,1,0,0,1,1,0,0], np.array([1,0,0,0,1,1,1,0]),
                         np.array([False]*4 + [True]*4))
        self.assertEqual(result['accuracy'], 0.75)
        self.assertEqual(result['eop_gap'], 0.5)
        self.assertEqual(result['selection_gap'], 0.5)

    def test_identifiers_and_labels_never_become_features(self):
        row = {name: [1] for name in NUM}
        row.update({name: ['category'] for name in CAT})
        row.update({'id_candidat':['C000001'], 'decision_octroi':[1], 'groupe':['remote'], 'strat':['remote_1']})
        x = features(pd.DataFrame(row))
        self.assertFalse({'id_candidat','decision_octroi','groupe','strat'} & set(x.columns))

    def test_merit_score_is_invariant_to_geographic_features(self):
        training = pd.read_csv(ROOT / 'data/donnees_demandes.csv').iloc[:500]
        model = dict(models())['logistic_C1.0']
        model.fit(features(training), training.decision_octroi)
        pair = pd.concat([training.iloc[[0]], training.iloc[[0]]], ignore_index=True)
        pair.loc[0, ['region_administrative', 'code_postal_3', 'distance_domicile_campus_km']] = ['Montreal', 'H1A', 5.0]
        pair.loc[1, ['region_administrative', 'code_postal_3', 'distance_domicile_campus_km']] = ['Cote-Nord', 'G4R', 300.0]
        scores = neutral_scores(model, pair)
        self.assertAlmostEqual(scores[0], scores[1], places=10)


if __name__ == '__main__':
    unittest.main()
