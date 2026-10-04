import unittest

import numpy as np
import pandas as pd

from model_effort import effort_predictions
from model_family_income import HOURS_WEIGHT, family_income_predictions, family_income_scores, describe


class FamilyIncomePolicyTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'id_candidat':[f'C{i:06}' for i in range(10)],
            'cote_r_equivalent':[30.] * 10, 'heures_travail_semaine':[10.] * 10,
            'revenu_familial_estime':[100000,90000,80000,70000,60000,50000,40000,30000,20000,10000]})

    def test_comparable_merit_prioritizes_lower_family_income(self):
        self.assertEqual(family_income_predictions(self.df).tolist(), [0,0,0,0,0,0,1,1,1,1])

    def test_income_doubling_has_modest_consistent_effect(self):
        sample=self.df.iloc[:3].copy()
        sample['revenu_familial_estime']=[30000.,60000.,120000.]
        scores=family_income_scores(sample)
        self.assertAlmostEqual(scores[0]-scores[1], .4644086109751634)
        self.assertAlmostEqual(scores[1]-scores[2], .4644086109751634)

    def test_reducing_income_cannot_remove_an_existing_award(self):
        original=family_income_predictions(self.df)
        for i in np.flatnonzero(original):
            changed=self.df.copy()
            changed.loc[i,'revenu_familial_estime'] /= 2
            self.assertEqual(family_income_predictions(changed)[i],1)

    def test_row_order_and_geography_do_not_change_decisions(self):
        expected=pd.Series(family_income_predictions(self.df),index=self.df.id_candidat)
        shuffled=self.df.assign(region_administrative='Cote-Nord',code_postal_3='G4R').sample(frac=1,random_state=42)
        actual=pd.Series(family_income_predictions(shuffled),index=shuffled.id_candidat)
        pd.testing.assert_series_equal(expected.sort_index(),actual.sort_index())

    def test_zero_weight_preserves_untied_effort_ranking(self):
        sample=self.df.copy()
        sample['cote_r_equivalent']=np.arange(10,dtype=float)+20
        np.testing.assert_array_equal(family_income_predictions(sample,0),effort_predictions(sample,HOURS_WEIGHT))

    def test_invalid_policy_weight_and_income_rejected(self):
        for weight in [-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):family_income_predictions(self.df,weight)
        for income in [0,-1,np.nan,np.inf]:
            changed=self.df.astype({'revenu_familial_estime':float})
            changed.loc[0,'revenu_familial_estime']=income
            with self.assertRaises(ValueError):family_income_predictions(changed)

    def test_empty_change_group_has_serializable_null_statistics(self):
        result=describe(self.df,np.zeros(len(self.df),dtype=bool))
        self.assertEqual(result['count'],0)
        self.assertTrue(all(v is None for v in result['median'].values()))


if __name__=='__main__':
    unittest.main()
