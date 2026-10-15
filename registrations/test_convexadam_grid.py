"""Physical geometry checks for the anisotropic ConvexAdam working grid."""
import unittest
import numpy as np
import SimpleITK as sitk
from run_convexadam import work_reference, displacement_to_lps


class ConvexAdamGridTests(unittest.TestCase):
    def test_native_xy_and_z_spacing_cover_fixed(self):
        fixed = sitk.Image([512, 512, 58], sitk.sitkFloat32)
        fixed.SetSpacing([0.9765625, 0.9765625, 5.])
        fixed.SetOrigin([-249.5, -249.5, -1031.3])
        work = work_reference(fixed, [*fixed.GetSpacing()[:2], 2.5], 6, 1)
        self.assertEqual(work.GetSpacing(), (0.9765625, 0.9765625, 2.5))
        self.assertEqual(work.GetSize(), (516, 516, 120))
        self.assertEqual(work.GetOrigin(), fixed.GetOrigin())
        self.assertTrue(np.all((np.array(work.GetSize())-1)*work.GetSpacing() >=
                               (np.array(fixed.GetSize())-1)*fixed.GetSpacing()))

    def test_padding_satisfies_both_pooling_strides(self):
        fixed = sitk.Image([19, 23, 27], sitk.sitkFloat32)
        work = work_reference(fixed, [.98, .98, 2.5], 6, 4)
        self.assertTrue(all(n % 6 == 0 and n % 4 == 0 for n in work.GetSize()))

    def test_anisotropic_displacement_units_and_direction(self):
        work = sitk.Image([4]*3, sitk.sitkFloat32)
        work.SetSpacing([.98, .98, 2.5])
        work.SetDirection([0., -1., 0., 1., 0., 0., 0., 0., 1.])
        # Z,Y,X = 2,3,4 voxels -> axis-mm = 3.92,2.94,5 -> LPS = -2.94,3.92,5.
        disp = np.array([[[[2., 3., 4.]]]])
        np.testing.assert_allclose(displacement_to_lps(disp, work), [[[[-2.94, 3.92, 5.]]]])


if __name__ == '__main__':
    unittest.main()
