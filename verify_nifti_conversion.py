"""Independent SimpleITK readback and source integrity checks for organized exports."""
import argparse
import json
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from convert_dicom_to_nifti import sha256


def verify(source, output):
    results = []
    for patient in sorted(output.iterdir()):
        if not (patient / 'index.json').is_file():
            continue
        index = json.loads((patient / 'index.json').read_text(encoding='utf-8'))
        records = {r['sop_uid']: r for r in index['sources']}
        for r in records.values():
            assert sha256(source / patient.name / r['path']) == r['sha256'], 'Source changed'
        for r in index['outputs']:
            assert sha256(patient / r['path']) == r['sha256'], 'Output changed'
        ct_images = {}
        for entry in index['images']:
            meta = json.loads((patient / entry['json']).read_text(encoding='utf-8'))
            img = sitk.ReadImage(str(patient / entry['nifti']))
            arr = sitk.GetArrayFromImage(img)
            for k, info in enumerate(meta['slices_in_output_order']):
                record = records[info['sop_uid']]
                ds = pydicom.dcmread(source / patient.name / record['path'])
                expected = ds.pixel_array.astype(np.float32) * float(ds.get('RescaleSlope', 1)) + float(ds.get('RescaleIntercept', 0))
                np.testing.assert_array_equal(arr[k], expected)
                for x, y in [(0, 0), (ds.Columns - 1, ds.Rows - 1)]:
                    o = np.asarray(ds.ImageOrientationPatient, float)
                    point = np.asarray(ds.ImagePositionPatient, float) + o[:3] * x * float(ds.PixelSpacing[1]) + o[3:] * y * float(ds.PixelSpacing[0])
                    np.testing.assert_allclose(img.TransformIndexToPhysicalPoint((x, y, k)), point, atol=.002)
            ct_images[entry['series_uid']] = img
        masks = 0
        for struct in index['structures']:
            for roi in struct['rois']:
                if roi['status'] != 'converted':
                    continue
                img = sitk.ReadImage(str(patient / roi['nifti']))
                ct = ct_images[roi['reference_ct_series_uid']]
                assert img.GetSize() == ct.GetSize()
                for getter in ('GetOrigin', 'GetSpacing', 'GetDirection'):
                    np.testing.assert_allclose(getattr(img, getter)(), getattr(ct, getter)(), atol=1e-5)
                values = sitk.GetArrayFromImage(img)
                assert set(np.unique(values)) <= {0, 1}
                assert int(values.sum()) == roi['voxel_count']
                masks += 1
        doses = 0
        for entry in index['doses']:
            if entry['status'] != 'converted':
                continue
            record = records[entry['sop_uid']]
            ds = pydicom.dcmread(source / patient.name / record['path'])
            img = sitk.ReadImage(str(patient / entry['nifti']))
            expected = ds.pixel_array.astype(np.float32) * float(ds.DoseGridScaling)
            np.testing.assert_array_equal(sitk.GetArrayFromImage(img), expected)
            orientation = np.asarray(ds.ImageOrientationPatient, float)
            origin = np.asarray(ds.ImagePositionPatient, float)
            offsets = np.asarray(ds.GridFrameOffsetVector, float)
            if not np.isclose(offsets[0], 0):
                offsets = offsets - origin[2]
            for k in range(int(ds.NumberOfFrames)):
                x, y = ds.Columns - 1, ds.Rows - 1
                point = (origin + orientation[:3] * x * float(ds.PixelSpacing[1])
                         + orientation[3:] * y * float(ds.PixelSpacing[0])
                         + np.cross(orientation[:3], orientation[3:]) * offsets[k])
                np.testing.assert_allclose(img.TransformIndexToPhysicalPoint((x, y, k)), point, atol=.002)
            doses += 1
        result = dict(patient=patient.name, source_files=len(records), ct=len(ct_images), masks=masks, doses=doses, verified=True)
        results.append(result)
        print(json.dumps(result), flush=True)
    assert results, 'No converted patients found'
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    verify(args.source, args.output)
