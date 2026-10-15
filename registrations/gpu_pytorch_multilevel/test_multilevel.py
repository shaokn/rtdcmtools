import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import torch
import SimpleITK as sitk
from airlab.transformation.pairwise import BsplineTransformation
from multilevel import PhysicalBspline, work_reference


class MultilevelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fixed = sitk.Image([25, 29, 33], sitk.sitkFloat32)
        self.fixed.SetSpacing([1., 1.2, 1.5])

    def test_full_resolution_matches_airlab_and_gradient(self):
        model = PhysicalBspline(self.fixed, self.fixed, 6., 'cpu')
        ref = BsplineTransformation(self.fixed.GetSize()[::-1], sigma=model.strides, order=3)
        torch.manual_seed(17)
        with torch.no_grad():
            model.coefficients.normal_(std=.1)
            ref.trans_parameters.copy_(model.coefficients)
        actual, expected = model(), ref()
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)
        g1 = torch.autograd.grad(actual.square().sum(), model.coefficients)[0]
        g2 = torch.autograd.grad(expected.square().sum(), ref.trans_parameters)[0]
        torch.testing.assert_close(g1, g2, rtol=5e-5, atol=2e-5)

    def test_coefficient_transfer_preserves_physical_field(self):
        models = [PhysicalBspline(self.fixed, work_reference(self.fixed, f), 6., 'cpu')
                  for f in (4, 2, 1)]
        torch.manual_seed(19)
        with torch.no_grad():
            models[0].coefficients.normal_(std=.2)
        for model in models[1:]:
            model.load_state_dict(models[0].state_dict())
        full = models[-1]().detach().numpy()
        for model, factor in zip(models[:-1], (4, 2)):
            # Sizes minus one divisible by four: all coarse voxels are full-grid samples.
            np.testing.assert_allclose(model().detach().numpy(), full[::factor, ::factor, ::factor],
                                       rtol=3e-5, atol=2e-6)

    def test_fractional_work_grid_is_same_interpolated_field(self):
        fixed = sitk.Image([24, 26, 28], sitk.sitkFloat32)
        coarse_ref = work_reference(fixed, 4)
        coarse = PhysicalBspline(fixed, coarse_ref, 6., 'cpu')
        full = PhysicalBspline(fixed, fixed, 6., 'cpu')
        with torch.no_grad():
            full.coefficients.normal_(std=.1)
        coarse.load_state_dict(full.state_dict())
        a = sitk.GetImageFromArray(full().detach().numpy().astype('float64'), isVector=True)
        a.CopyInformation(fixed)
        sampled = sitk.Resample(a, coarse_ref, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear)
        np.testing.assert_allclose(coarse().detach().numpy(), sitk.GetArrayFromImage(sampled),
                                   rtol=4e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
