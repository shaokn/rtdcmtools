import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, RTDoseStorage, generate_uid
from PIL import Image

import server


class GeometryTests(unittest.TestCase):
    def test_filled_isodose_bands(self):
        arr = np.zeros((4, 20, 40), dtype=np.float32)
        image = server.image_from_array(arr, [0, 0, 0], [1, 1, 1], np.eye(3))
        dose = arr.copy()
        dose[:, :, 10:20] = 7
        dose[:, :, 20:30] = 12
        dose[:, :, 30:] = np.nan
        fake = SimpleNamespace(array=arr, image=image, dose=lambda uid: (dose, 20))
        client = server.app.test_client()
        params = {'case': 'test', 'series': 'test', 'axis': 'axial', 'index': 1,
                  'dose': 'test', 'dose_mode': 'filled', 'opacity': .5, 'levels': '10,5,10'}
        with patch.object(server, 'volume', return_value=fake):
            response = client.get('/api/slice', query_string=params)
            self.assertEqual(response.status_code, 200)
            pixels = np.asarray(Image.open(io.BytesIO(response.data)))
            for x in [112, 787]:
                np.testing.assert_array_equal(pixels[225, x], [102, 102, 102])
            for x, level in [(337, 5), (562, 10)]:
                expected = (102 * .5 + server.color_dose(np.array([level]), 20)[0] * .5).astype('uint8')
                np.testing.assert_allclose(pixels[225, x], expected, atol=1)
            for levels in ['', '-5', 'nan']:
                self.assertEqual(client.get('/api/slice', query_string={**params, 'levels': levels}).status_code, 400)

    def test_line_opacity_zero_matches_ct(self):
        arr = np.zeros((4, 8, 8), dtype=np.float32)
        image = server.image_from_array(arr, [0, 0, 0], [1, 1, 1], np.eye(3))
        dose = arr.copy()
        dose[:, 2:6, 2:6] = 10
        fake = SimpleNamespace(array=arr, image=image, dose=lambda uid: (dose, 10))
        client = server.app.test_client()
        params = {'case': 'test', 'series': 'test', 'axis': 'axial', 'index': 1}
        with patch.object(server, 'volume', return_value=fake):
            baseline = client.get('/api/slice', query_string=params).data
            hidden = client.get('/api/slice', query_string={**params, 'dose': 'test',
                'dose_mode': 'lines', 'opacity': 0, 'levels': '5'}).data
            visible = client.get('/api/slice', query_string={**params, 'dose': 'test',
                'dose_mode': 'lines', 'opacity': 1, 'levels': '5'}).data
        self.assertEqual(baseline, hidden)
        self.assertNotEqual(baseline, visible)

    def test_dose_axis_spacing_origin_scaling(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'dose.dcm'
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = RTDoseStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
            ds.SOPClassUID = RTDoseStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.Rows, ds.Columns, ds.NumberOfFrames = 3, 4, 2
            ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit = 1, 16, 16, 15
            ds.PixelRepresentation = 0
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.ImageOrientationPatient = [0, 1, 0, -1, 0, 0]
            ds.ImagePositionPatient = [11, 22, 33]
            ds.PixelSpacing = [2, 3]
            ds.GridFrameOffsetVector = [0, -5]
            ds.DoseGridScaling, ds.DoseUnits = .1, 'GY'
            arr = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
            ds.PixelData = arr.tobytes()
            ds.save_as(path, enforce_file_format=True)
            image, _, maximum = server.load_dose(path)
            self.assertEqual(image.GetSize(), (4, 3, 2))
            np.testing.assert_allclose(image.TransformIndexToPhysicalPoint((2, 1, 1)), [9, 28, 28])
            self.assertAlmostEqual(image[2, 1, 1], 1.8, places=5)
            self.assertAlmostEqual(maximum, 2.3, places=5)

    def test_all_five_cases_and_roi_projection(self):
        client = server.app.test_client()
        cases = client.get('/api/cases').json
        self.assertEqual(len(cases), 5)
        result = []
        for case in cases:
            info = client.get('/api/info', query_string={'case': case['id']}).json
            series = info['series'][0]['uid']
            struct = info['structures'][0]
            v = server.volume(case['id'], series)
            # The rt-utils raster coordinates must agree with DICOM contour physical coordinates.
            heart = next(r for r in struct['rois'] if r['name'] == 'Heart')
            mask = v.mask(struct['uid'], heart['number'])
            points = np.argwhere(mask)
            ds = pydicom.dcmread(server.path_for(case['id'], v.record(struct['uid'], 'RTSTRUCT')))
            contour = next(c for c in ds.ROIContourSequence if c.ReferencedROINumber == heart['number'])
            physical = np.concatenate([np.asarray(c.ContourData).reshape(-1, 3) for c in contour.ContourSequence])
            indices = np.array([v.image.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in physical])
            np.testing.assert_allclose(points.min(axis=0)[::-1], indices.min(axis=0), atol=2)
            np.testing.assert_allclose(points.max(axis=0)[::-1], indices.max(axis=0), atol=2)
            maxima = []
            for d in info['doses']:
                response = client.get('/api/dose_meta', query_string={'case': case['id'], 'series': series, 'dose': d['uid']})
                self.assertEqual(response.status_code, 200, response.json)
                maxima.append(response.json['maximum'])
            dose = next(d for d in info['doses'] if d['type'] == 'PLAN')
            for axis, dim in [('axial', 2), ('coronal', 1), ('sagittal', 0)]:
                response = client.get('/api/slice', query_string={'case': case['id'], 'series': series,
                    'axis': axis, 'index': v.image.GetSize()[dim] // 2, 'struct': struct['uid'],
                    'rois': heart['number'], 'dose': dose['uid']})
                self.assertEqual(response.status_code, 200)
                pixels = np.asarray(Image.open(io.BytesIO(response.data)))
                self.assertGreater(pixels.std(), 20)
            result.append({'case': case['id'], 'dose_maxima_Gy': maxima,
                           'heart_voxels': len(points), 'three_planes': 'passed'})
        (server.HERE / 'qa').mkdir(exist_ok=True)
        (server.HERE / 'qa' / 'data_checks.json').write_text(json.dumps(result, indent=2))

    def test_error_response(self):
        client = server.app.test_client()
        self.assertEqual(client.get('/api/info?case=../../etc').status_code, 400)


if __name__ == '__main__':
    unittest.main()
