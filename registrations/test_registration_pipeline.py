import unittest
import numpy as np
import SimpleITK as s
from registration_pipeline import same_grid
from registration_utils import map_image


class OriginalSamplingTests(unittest.TestCase):
    def test_direct_label_sampling_avoids_double_nearest_rounding(self):
        a=np.zeros((12,12,12),dtype='uint8');a[6,6,6]=1
        mask=s.GetImageFromArray(a)
        rigid=s.TranslationTransform(3,[.4,0,0])
        residual=s.TranslationTransform(3,[.4,0,0])
        total=s.CompositeTransform([rigid,residual])
        direct=map_image(mask,mask,total,True)
        staged=map_image(map_image(mask,mask,rigid,True),mask,residual,True)
        # A pull map x+0.8 samples source x=6 at output x=5.
        self.assertEqual(int(s.GetArrayFromImage(direct)[6,6,5]),1)
        self.assertEqual(int(s.GetArrayFromImage(staged)[6,6,6]),1)
        self.assertTrue(same_grid(direct,mask))
        self.assertFalse(np.array_equal(s.GetArrayFromImage(direct),s.GetArrayFromImage(staged)))

    def test_total_pull_order_with_rotation(self):
        rigid=s.AffineTransform(3);rigid.SetMatrix([0.,-1.,0.,1.,0.,0.,0.,0.,1.])
        rigid.SetTranslation([5.,-2.,1.])
        residual=s.TranslationTransform(3,[1.,2.,3.])
        total=s.CompositeTransform([rigid,residual])
        np.testing.assert_allclose(total.TransformPoint([0.,0.,0.]),[3.,-1.,4.])


if __name__=='__main__':unittest.main()
