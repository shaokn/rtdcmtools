"""Synthetic geometry tests; no patient data required."""
import tempfile
from pathlib import Path
import unittest

import nibabel as nib
import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

import convert_dicom_to_nifti as c


def image(array, position=(10, 20, 30), orientation=(1, 0, 0, 0, 1, 0)):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Rows, ds.Columns = array.shape[-2:]
    ds.ImagePositionPatient = list(position)
    ds.ImageOrientationPatient = list(orientation)
    ds.PixelSpacing = [2, 3]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.asarray(array, dtype='<u2').tobytes()
    ds.SeriesInstanceUID = '1.2.3'
    ds.FrameOfReferenceUID = '1.2.4'
    ds.SOPInstanceUID = '1.2.5'
    if array.ndim == 3:
        ds.NumberOfFrames = array.shape[0]
    return ds


class ConversionTests(unittest.TestCase):
    def test_ct_sort_hu_and_axes(self):
        a = image(np.arange(12).reshape(3, 4))
        b = image(np.arange(12).reshape(3, 4) + 100, position=(10, 20, 35))
        for ds in (a, b):
            ds.RescaleSlope = 2
            ds.RescaleIntercept = -1000
        values, affine, ordered = c.ct_volume([b, a])
        self.assertIs(ordered[0], a)
        self.assertEqual(values.shape, (4, 3, 2))
        self.assertEqual(values[2, 1, 1], -788)
        np.testing.assert_allclose(c.LPS_TO_RAS @ affine @ [2, 1, 1, 1], [16, 22, 35, 1])
        b.ImagePositionPatient = [11, 20, 35]
        with self.assertRaises(ValueError):
            c.ct_volume([a, b])

    def test_dose_oblique_descending_and_scaling(self):
        ds = image(np.arange(24).reshape(2, 3, 4), orientation=(0, 1, 0, 0, 0, 1))
        ds.DoseUnits = 'GY'
        ds.DoseGridScaling = .01
        ds.GridFrameOffsetVector = [0, -5]
        values, affine, _ = c.dose_volume(ds)
        self.assertAlmostEqual(float(values[2, 1, 1]), .18, places=6)
        np.testing.assert_allclose(c.LPS_TO_RAS @ affine @ [2, 1, 1, 1], [5, 26, 32, 1])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'dose.nii.gz'
            c.write_nifti(path, values, affine)
            self.assertAlmostEqual(float(nib.load(path).get_fdata()[2, 1, 1]), .18, places=6)
        ds.DoseUnits = 'RELATIVE'
        with self.assertRaises(ValueError):
            c.dose_volume(ds)

    def test_absolute_z_and_nonuniform(self):
        ds = image(np.arange(36).reshape(3, 3, 4))
        ds.DoseUnits = 'GY'
        ds.DoseGridScaling = .1
        ds.GridFrameOffsetVector = [30, 35, 40]
        _, affine, convention = c.dose_volume(ds)
        self.assertEqual(convention, 'absolute_patient_z')
        self.assertEqual(affine[2, 3], 30)
        ds.GridFrameOffsetVector = [30, 35, 41]
        with self.assertRaises(ValueError):
            c.dose_volume(ds)

    def test_mask_alignment_and_explicit_clipping(self):
        slices = [image(np.zeros((16, 16)), position=(0, 0, z)) for z in (0, 2)]
        for k, ds in enumerate(slices):
            ds.SOPInstanceUID = f'1.2.5.{k}'
            ds.PixelSpacing = [1, 1]
        arr, affine, ordered = c.ct_volume(slices)
        ct = dict(datasets=ordered, affine=affine, shape=arr.shape)
        contour = Dataset()
        contour.ContourGeometricType = 'CLOSED_PLANAR'
        contour.NumberOfContourPoints = 4
        contour.ContourData = [2, 7, 0, 4, 7, 0, 4, 10, 0, 2, 10, 0]
        ref = Dataset()
        ref.ReferencedSOPInstanceUID = slices[0].SOPInstanceUID
        contour.ContourImageSequence = [ref]
        mask, info = c.rasterize_roi(ct, [contour], '1.2.4')
        self.assertEqual(mask[3, 8, 0], 1)
        self.assertEqual(mask[8, 3, 0], 0)
        self.assertEqual(mask.sum(), 12)
        self.assertFalse(info['clipped_to_ct_grid'])
        contour.ContourData = [-2, 7, 0, 4, 7, 0, 4, 10, 0, -2, 10, 0]
        with self.assertRaises(ValueError):
            c.rasterize_roi(ct, [contour], '1.2.4')
        mask, info = c.rasterize_roi(ct, [contour], '1.2.4', True)
        self.assertTrue(info['clipped_to_ct_grid'])
        self.assertEqual(mask.sum(), 20)


if __name__ == '__main__':
    unittest.main()
