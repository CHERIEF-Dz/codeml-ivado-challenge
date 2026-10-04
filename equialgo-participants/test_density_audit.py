import unittest

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from audit_density import density_features, transfer_dbscan


class DensityAuditTests(unittest.TestCase):
    def test_labels_and_sensitive_categories_do_not_enter_distances(self):
        row={'cote_r_equivalent':[28.], 'heures_travail_semaine':[10.],
             'revenu_familial_estime':[60000.], 'distance_domicile_campus_km':[30.],
             'id_candidat':['C000001'], 'decision_octroi':[0], 'region_administrative':['Montreal'],
             'code_postal_3':['H1A'], 'programme_etudes':['Genie']}
        original=pd.DataFrame(row)
        changed=original.assign(id_candidat='C999999',decision_octroi=1,region_administrative='Cote-Nord',code_postal_3='G4R')
        pd.testing.assert_frame_equal(density_features(original),density_features(changed))
        self.assertEqual(list(density_features(original).columns),['r_score','work_hours','log_income','log_distance'])

    def test_nearest_core_transfer_and_ambiguous_border(self):
        train=np.array([[-1.],[-1.],[-1.],[1.],[1.],[1.]])
        model=DBSCAN(eps=1.,min_samples=3).fit(train)
        labels,ambiguous=transfer_dbscan(model,np.array([[-.8],[0.],[.8],[4.]]))
        self.assertEqual(labels[0],model.labels_[0])
        self.assertEqual(labels[2],model.labels_[3])
        self.assertEqual(labels[3],-1)
        self.assertIn(labels[1],[model.labels_[0],model.labels_[3]])
        self.assertEqual(ambiguous.tolist(),[False,True,False,False])

    def test_model_with_no_core_labels_all_queries_noise(self):
        model=DBSCAN(eps=.1,min_samples=3).fit(np.array([[0.],[10.]]))
        labels,ambiguous=transfer_dbscan(model,np.array([[0.],[5.]]))
        self.assertEqual(labels.tolist(),[-1,-1])
        self.assertFalse(ambiguous.any())


if __name__=='__main__':
    unittest.main()
