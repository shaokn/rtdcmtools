"""Custom absolute-dose DVH metrics served by server_new.py.

These tests are self-contained: the metric engine is pure, and the route test
patches the volume accessor instead of reading case data from disk.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import server_new


class ParseMetricTests(unittest.TestCase):
    def test_empty_spec_keeps_the_legacy_columns(self):
        self.assertEqual([m[0] for m in server_new.parse_metrics('')],
                         ['Dmean', 'D95', 'D2', 'V20Gy'])

    def test_supported_tokens(self):
        metrics = server_new.parse_metrics('d98%, D2cc, v20, Dmax, dmean, min, VOL')
        self.assertEqual([m[0] for m in metrics],
                         ['D98', 'D2cc', 'V20Gy', 'Dmax', 'Dmean', 'Dmin', 'Volume'])
        self.assertEqual([m[1] for m in metrics],
                         ['dx', 'dxcc', 'vx', 'max', 'mean', 'min', 'volume'])

    def test_duplicate_tokens_collapse(self):
        self.assertEqual([m[0] for m in server_new.parse_metrics('D95%, d95%, D95%')], ['D95'])

    def test_relative_tokens_explain_what_is_missing(self):
        with self.assertRaises(ValueError) as caught:
            server_new.parse_metric('V107%')
        self.assertIn('处方剂量', str(caught.exception))

    def test_dx_percent_spelling_is_the_dose_to_that_volume(self):
        # ``D95%`` spells the volume share, not a share of the prescription, so
        # it needs no reference dose.
        percent = server_new.parse_metric('D95%')
        self.assertEqual(percent[0], 'D95')
        self.assertEqual(percent[1:3], ('dx', 95.0))
        self.assertEqual((percent[3], percent[4]), ('D95%', 'Gy'))
        self.assertEqual([m[0] for m in server_new.parse_metrics('D98%, d2%, D2cc')],
                         ['D98', 'D2', 'D2cc'])
        self.assertEqual(server_new.parse_metric('D2%'), server_new.parse_metric('d2%'))

    def test_bare_dx_is_refused_with_the_canonical_spelling(self):
        for token in ('D95', 'd2', 'D50'):
            with self.assertRaises(ValueError) as caught:
                server_new.parse_metric(token)
            self.assertIn(f'{token.upper()}%', str(caught.exception))

    def test_invalid_tokens_are_rejected(self):
        for token in ('D0', 'D100', 'D0%', 'D100%', 'V-1', 'D95Gy', 'D95ccGy',
                      'D95%%', 'foobar', 'D1e3'):
            with self.assertRaises(ValueError, msg=token):
                server_new.parse_metric(token)

    def test_metric_count_is_capped(self):
        with self.assertRaises(ValueError):
            server_new.parse_metrics(','.join(f'D{i}%' for i in range(1, 14)))


class EvaluateMetricTests(unittest.TestCase):
    def setUp(self):
        self.values = np.sort(np.arange(10, dtype=np.float32))

    def test_dx_uses_the_same_estimator_as_the_builtin_columns(self):
        self.assertEqual(server_new.evaluate_metric('dx', 95, self.values, 1.0),
                         float(np.percentile(self.values, 5)))
        self.assertEqual(server_new.evaluate_metric('dx', 2, self.values, 1.0),
                         float(np.percentile(self.values, 98)))

    def test_dxcc_is_the_lowest_dose_inside_the_hottest_volume(self):
        # 3 cc at 1 cc per voxel covers the three hottest voxels 9, 8, 7.
        self.assertEqual(server_new.evaluate_metric('dxcc', 3, self.values, 1.0), 7.0)

    def test_dxcc_below_one_voxel_falls_back_to_the_hottest_voxel(self):
        self.assertEqual(server_new.evaluate_metric('dxcc', 0.1, self.values, 1.0), 9.0)
        self.assertTrue(server_new.metric_notes('dxcc', 0.1, self.values, 1.0))

    def test_dxcc_wider_than_the_roi_reports_dmin(self):
        self.assertEqual(server_new.evaluate_metric('dxcc', 50, self.values, 1.0), 0.0)
        self.assertTrue(server_new.metric_notes('dxcc', 50, self.values, 1.0))

    def test_vx_is_the_percent_volume_at_or_above_the_level(self):
        self.assertEqual(server_new.evaluate_metric('vx', 5, self.values, 1.0), 50.0)

    def test_scalar_metrics(self):
        self.assertEqual(server_new.evaluate_metric('mean', None, self.values, 1.0), 4.5)
        self.assertEqual(server_new.evaluate_metric('max', None, self.values, 1.0), 9.0)
        self.assertEqual(server_new.evaluate_metric('min', None, self.values, 1.0), 0.0)
        self.assertEqual(server_new.evaluate_metric('volume', None, self.values, 0.5), 5.0)

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            server_new.evaluate_metric('d97', 97, self.values, 1.0)


class DvhRouteTests(unittest.TestCase):
    """The override must replace the core endpoint rather than shadow it."""

    def setUp(self):
        self.client = server_new.core.app.test_client()
        array = np.zeros((4, 8, 8), dtype=np.float32)
        dose = np.tile(np.arange(8, dtype=np.float32)[None, :, None], (4, 1, 8))
        mask = np.zeros((4, 8, 8), dtype=bool)
        mask[1:3, 2:6, 1:4] = True
        self.dose, self.mask = dose, mask
        self.fake = SimpleNamespace(
            image=server_new.core.image_from_array(array, [0, 0, 0], [2, 2, 2], np.eye(3)),
            dose=lambda uid: (dose, float(dose.max())),
            record=lambda uid, modality: {'rois': [{'number': 1, 'name': 'PTV', 'clipped': False}]},
            mask=lambda uid, number: mask,
        )

    def test_endpoint_is_replaced_not_duplicated(self):
        registered = server_new.core.app.view_functions['dvh']
        self.assertIs(getattr(registered, '__wrapped__', None), server_new.dvh_with_metrics)
        rules = [rule for rule in server_new.core.app.url_map.iter_rules() if str(rule) == '/api/dvh']
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].endpoint, 'dvh')

    def request(self, **overrides):
        params = {'case': 'test', 'series': 'test', 'dose': 'test', 'struct': 'test', 'rois': '1'}
        params.update(overrides)
        with patch.object(server_new.core, 'volume', return_value=self.fake):
            return self.client.get('/api/dvh', query_string=params)

    def test_requested_columns_and_values(self):
        payload = self.request(metrics='D50%,D1cc,V5Gy,Dmax,volume').json
        self.assertEqual([m['key'] for m in payload['metrics']],
                         ['D50', 'D1cc', 'V5Gy', 'Dmax', 'Volume'])
        row = payload['structures'][0]
        values = self.dose[self.mask]
        self.assertEqual(row['name'], 'PTV')
        self.assertAlmostEqual(row['volume_cc'], len(values) * 0.008, places=6)
        self.assertAlmostEqual(row['values']['D50'], float(np.percentile(values, 50)), places=6)
        self.assertAlmostEqual(row['values']['V5Gy'], float((values >= 5).mean() * 100), places=6)
        self.assertAlmostEqual(row['values']['Dmax'], float(values.max()), places=6)
        self.assertAlmostEqual(row['values']['Volume'], len(values) * 0.008, places=6)
        # 1 cc is far wider than this 0.192 cc ROI, so D1cc collapses to Dmin.
        self.assertAlmostEqual(row['values']['D1cc'], float(values.min()), places=6)
        self.assertTrue(row['notes'])

    def test_percent_spelling_returns_the_same_number(self):
        payload = self.request(metrics='D95%').json
        self.assertEqual([m['key'] for m in payload['metrics']], ['D95'])
        self.assertEqual([m['label'] for m in payload['metrics']], ['D95%'])
        row = payload['structures'][0]
        # The canonical % form must land on the built-in D95 estimator, not on
        # anything that would need a prescription dose.
        self.assertAlmostEqual(row['values']['D95'],
                               float(np.percentile(self.dose[self.mask], 5)), places=6)
        self.assertAlmostEqual(row['values']['D95'], row['d95'], places=6)

    def test_bare_dx_is_a_readable_400(self):
        response = self.request(metrics='D95')
        self.assertEqual(response.status_code, 400)
        self.assertIn('D95%', response.json['error'])

    def test_legacy_keys_survive_for_the_old_front_end(self):
        row = self.request().json['structures'][0]
        for key in ('mean', 'd95', 'd2', 'v20', 'curve', 'coverage'):
            self.assertIn(key, row)
        self.assertEqual(len(row['curve']), 201)
        self.assertAlmostEqual(row['d95'], float(np.percentile(self.dose[self.mask], 5)), places=6)

    def test_bad_metric_returns_a_readable_400(self):
        response = self.request(metrics='V107%')
        self.assertEqual(response.status_code, 400)
        self.assertIn('处方剂量', response.json['error'])

    def test_csv_export_follows_the_requested_columns(self):
        response = self.request(metrics='D98%,D2cc', format='csv')
        self.assertEqual(response.status_code, 200)
        lines = response.get_data(as_text=True).lstrip('\ufeff').strip().splitlines()
        self.assertEqual(lines[0], 'ROI,volume_cc,coverage,D98%_Gy,D2cc_Gy,note')
        self.assertEqual(len(lines), 2)

    def test_legacy_default_spec_still_matches_the_fixed_columns(self):
        payload = self.request().json
        self.assertEqual([m['label'] for m in payload['metrics']],
                         ['Dmean', 'D95%', 'D2%', 'V20Gy'])
        row = payload['structures'][0]
        self.assertAlmostEqual(row['values']['D95'], row['d95'], places=6)
        self.assertAlmostEqual(row['values']['D2'], row['d2'], places=6)
        self.assertAlmostEqual(row['values']['V20Gy'], row['v20'], places=6)


if __name__ == '__main__':
    unittest.main()
