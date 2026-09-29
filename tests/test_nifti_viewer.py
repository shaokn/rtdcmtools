"""Cross-check NIfTI viewer against the organized DICOM on this workstation."""
import io
import unittest

import numpy as np
from PIL import Image
import server
import nifti_backend


class NiftiViewerTests(unittest.TestCase):
    def test_all_cases(self):
        client = server.app.test_client()
        cases = client.get('/api/cases?source=nifti').json
        self.assertEqual(len(cases), 5)
        for case in cases:
            info = client.get('/api/info', query_string=dict(source='nifti', case=case['id'])).json
            series = info['series'][0]['uid']
            n = nifti_backend.Volume(server.NIFTI_ROOT, case['id'], series)
            d = server.Volume(case['id'], series)
            np.testing.assert_array_equal(n.array, d.array)
            struct = info['structures'][0]
            heart = next(r for r in struct['rois'] if r['name'] == 'Heart')
            np.testing.assert_array_equal(n.mask(struct['uid'], heart['number']), d.mask(struct['uid'], heart['number']))
            self.assertEqual(len(info['doses']), 2)
            for dose in info['doses']:
                mapped, maximum = n.dose(dose['uid'])
                expected, refmax = d.dose(dose['uid'])
                np.testing.assert_allclose(mapped, expected, atol=.001, rtol=0, equal_nan=True)
                self.assertAlmostEqual(maximum, refmax, places=5)
            params = dict(source='nifti', case=case['id'], series=series,
                          struct=struct['uid'], rois=str(heart['number']), dose=info['doses'][0]['uid'])
            for axis, dim in [('axial', 2), ('coronal', 1), ('sagittal', 0)]:
                for mode in ('wash', 'lines', 'filled'):
                    r = client.get('/api/slice', query_string={**params, 'axis': axis,
                        'index': n.image.GetSize()[dim] // 2, 'dose_mode': mode})
                    self.assertEqual(r.status_code, 200, r.json)
                    self.assertGreater(np.asarray(Image.open(io.BytesIO(r.data))).std(), 20)
            r = client.get('/api/dvh', query_string=params)
            self.assertEqual(r.status_code, 200)
            self.assertIn('mean', r.json['structures'][0])
            ref = client.get('/api/dvh', query_string={**params, 'source': 'dicom'}).json['structures'][0]
            self.assertAlmostEqual(r.json['structures'][0]['mean'], ref['mean'], places=3)
            clipped = next(r for r in struct['rois'] if r['clipped'])
            r = client.get('/api/dvh', query_string={**params, 'rois': clipped['number']})
            self.assertNotIn('mean', r.json['structures'][0])
            self.assertIn('裁剪', r.json['structures'][0]['error'])
            r = client.get('/api/dvh', query_string={**params, 'format': 'csv'})
            self.assertEqual(r.status_code, 200)
            self.assertIn(b'Dmean_Gy', r.data)

    def test_safety(self):
        client = server.app.test_client()
        self.assertEqual(client.get('/api/cases?source=unknown').status_code, 400)
        self.assertEqual(client.get('/api/info?source=nifti&case=../../etc').status_code, 400)
        case = next(iter(nifti_backend.catalog(server.NIFTI_ROOT)))
        with self.assertRaises(ValueError):
            nifti_backend.path_for(server.NIFTI_ROOT, case, '../../outside.nii.gz')
        info = nifti_backend.info(server.NIFTI_ROOT, case)
        n = nifti_backend.Volume(server.NIFTI_ROOT, case, info['series'][0]['uid'])
        entry = next(e for e in n.index['doses'] if e['status'] == 'converted')
        entry['dose_units'] = 'RELATIVE'
        with self.assertRaisesRegex(ValueError, 'Gy'):
            n.dose(entry['sop_uid'])
        entry['dose_units'] = 'Gy'
        n.frame = 'mismatch'
        with self.assertRaisesRegex(ValueError, 'FrameOfReferenceUID'):
            n.dose(entry['sop_uid'])


if __name__ == '__main__':
    unittest.main()
