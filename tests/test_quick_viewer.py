"""Quick-view tests use synthetic NIfTI files, with no JSON sidecars."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
from PIL import Image

import server


class QuickTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.client = server.app.test_client()
        server.QUICK.clear()
        self.array = np.arange(8 * 12 * 16, dtype=np.float32).reshape(8, 12, 16) - 900
        self.ct = self.write('ct.nii.gz', self.array)
        mask = np.zeros_like(self.array, dtype=np.uint8)
        mask[1:5, 3:8, 2:6] = 1
        mask[4:7, 3:8, 9:13] = 2
        self.mask = self.write('mask.nii', mask)
        self.dose = self.write('dose.nii.gz', np.full_like(self.array, 2500))

    def tearDown(self):
        server.QUICK.clear()
        self.temp.cleanup()

    def write(self, name, array, origin=(10, 20, 30)):
        image = sitk.GetImageFromArray(array)
        image.SetOrigin(origin)
        image.SetSpacing((1, 2, 3))
        path = self.root / name
        sitk.WriteImage(image, str(path))
        return path

    def post(self, overlays=False, **changes):
        data = {'ct': (io.BytesIO(self.ct.read_bytes()), self.ct.name)}
        if overlays:
            data.update(masks=(io.BytesIO(self.mask.read_bytes()), self.mask.name),
                        dose=(io.BytesIO(self.dose.read_bytes()), self.dose.name),
                        same_space='yes', dose_units='cGy')
        data.update(changes)
        with patch.object(server, 'QUICK_TEMP', self.root / 'uploads'):
            response = self.client.post('/api/quick', data=data, headers={'X-Quick-View': '1'})
        self.assertFalse(list((self.root / 'uploads').glob('viewer-quick-*')))
        return response

    def test_ct_only_and_clear(self):
        r = self.post()
        self.assertEqual(r.status_code, 200, r.json)
        token = r.json['case']
        v = server.quick_volume(token)
        np.testing.assert_array_equal(v.array, self.array)
        self.assertEqual(v.info['doses'], [])
        for axis, index in [('axial', 3), ('coronal', 5), ('sagittal', 7)]:
            r = self.client.get('/api/slice', query_string=dict(source='quick', case=token,
                                series='ct', axis=axis, index=index))
            self.assertEqual(r.status_code, 200)
            self.assertGreater(np.asarray(Image.open(io.BytesIO(r.data))).std(), 1)
        self.assertEqual(self.client.get('/api/dvh?source=quick').status_code, 400)
        self.client.delete('/api/quick', query_string={'case': token}, headers={'X-Quick-View': '1'})
        self.assertEqual(self.client.get('/api/info', query_string=dict(source='quick', case=token)).status_code, 400)

    def test_labels_and_cgy(self):
        r = self.post(True)
        self.assertEqual(r.status_code, 200, r.json)
        v = server.quick_volume(r.json['case'])
        self.assertEqual(len(v.rois), 2)
        self.assertEqual(v.mask('masks', 1).sum(), 80)
        values, maximum = v.dose('dose')
        np.testing.assert_allclose(values, 25)
        self.assertEqual(maximum, 25)
        params = dict(source='quick', case=r.json['case'], series='ct', struct='masks',
                      rois='1,2', dose='dose', index=3, fill=1)
        for mode in ('wash', 'lines', 'filled'):
            response = self.client.get('/api/slice', query_string={**params, 'dose_mode': mode})
            self.assertEqual(response.status_code, 200, response.json)
        r = self.post(True, dose_units='Gy')
        self.assertEqual(server.quick_volume(r.json['case']).dose('dose')[1], 2500)

    def test_validation(self):
        self.assertEqual(self.post(True, dose_units='').status_code, 400)
        self.assertEqual(self.post(True, same_space='no').status_code, 400)
        self.assertEqual(self.post(ct=(io.BytesIO(b'bad'), 'broken.nii')).status_code, 400)
        self.assertEqual(self.post(ct=(io.BytesIO(b'bad'), 'other.txt')).status_code, 400)
        shifted = self.write('shifted.nii', np.ones_like(self.array, dtype=np.uint8), (20, 20, 30))
        r = self.post(True, masks=(io.BytesIO(shifted.read_bytes()), shifted.name))
        self.assertEqual(r.status_code, 400)
        self.assertIn('网格', r.json['error'])
        self.assertEqual(self.client.post('/api/quick').status_code, 400)
        with patch.dict(server.app.config, MAX_CONTENT_LENGTH=32):
            self.assertEqual(self.post().status_code, 413)

    def test_flips_expiry_and_eviction(self):
        image = sitk.ReadImage(str(self.ct))
        image = sitk.DICOMOrient(image, 'RAS')
        sitk.WriteImage(image, str(self.ct))
        token = self.post().json['case']
        np.testing.assert_array_equal(server.quick_volume(token).array, self.array)
        self.post()
        self.post()
        self.assertNotIn(token, server.QUICK)
        with patch.object(server.time, 'monotonic', return_value=10**12):
            self.client.get('/api/cases?source=quick')
        self.assertFalse(server.QUICK)


if __name__ == '__main__':
    unittest.main()
