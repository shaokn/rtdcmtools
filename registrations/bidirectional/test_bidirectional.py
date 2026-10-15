import unittest
import numpy as np
import SimpleITK as s
import torch
from run_bidirectional import normalized_query,sample,cycle_image,field_image,aligned_reference


class InverseConsistencyTests(unittest.TestCase):
    def test_rotated_common_grid_covers_original_ct(self):
        import itertools
        moving=s.Image([15,17,19],s.sitkFloat32);moving.SetOrigin([-5.,-6.,-7.]);moving.SetSpacing([1.,1.2,2.])
        rigid=s.Euler3DTransform();rigid.SetRotation(.04,-.03,.1);rigid.SetTranslation([2.,-1.,3.])
        ref=aligned_reference(moving,rigid)
        self.assertTrue(np.allclose(ref.GetDirection(),np.eye(3).ravel()))
        for index in itertools.product(*[(0,n-1) for n in moving.GetSize()]):
            point=rigid.GetInverse().TransformPoint(moving.TransformIndexToPhysicalPoint(index))
            idx=ref.TransformPhysicalPointToContinuousIndex(point)
            self.assertTrue(all(-1e-6<=v<=n-1+1e-6 for v,n in zip(idx,ref.GetSize())))

    def test_physical_query_between_different_origins_and_spacings(self):
        ref=s.Image([9,10,11],s.sitkFloat32);ref.SetOrigin([2.,3.,4.]);ref.SetSpacing([1.,1.5,2.])
        dest=s.Image([20,21,22],s.sitkFloat32);dest.SetOrigin([-2.,-3.,-4.]);dest.SetSpacing([2.,2.5,3.])
        d=torch.zeros((11,10,9,3));d[:]=torch.tensor([1.,2.,3.])
        q=normalized_query(ref,d,dest).numpy()
        expected=2*(np.array([3.,5.,7.])-np.array(dest.GetOrigin()))/((np.array(dest.GetSize())-1)*dest.GetSpacing())-1
        np.testing.assert_allclose(q[0,0,0,0],expected,atol=1e-6)

    def test_nonconstant_inverse_fields_roundtrip(self):
        ref=s.Image([15]*3,s.sitkFloat32);ref.SetOrigin([-7.]*3)
        z,y,x=np.mgrid[-7:8,-7:8,-7:8]
        pos=np.stack([x,y,z],axis=-1)
        u=.05*pos+np.array([.2,-.3,.1])
        v=-(.05*pos+np.array([.2,-.3,.1]))/1.05
        err,valid=cycle_image(field_image(u,ref),field_image(v,ref))
        self.assertLess(float(s.GetArrayFromImage(err)[s.GetArrayFromImage(valid)>0].max()),1e-5)

    def test_cycle_loss_backpropagates_into_both_directions(self):
        ref=s.Image([7]*3,s.sitkFloat32)
        u=torch.full((7,7,7,3),.1,requires_grad=True)
        v=torch.full((7,7,7,3),.2,requires_grad=True)
        q=normalized_query(ref,u,ref)
        back=sample(v.permute(3,0,1,2)[None],q,padding='border')[0].permute(1,2,3,0)
        loss=(u+back).square().mean()
        gu,gv=torch.autograd.grad(loss,[u,v])
        self.assertGreater(float(gu.abs().sum()),0)
        self.assertGreater(float(gv.abs().sum()),0)
        self.assertTrue(torch.isfinite(gu).all() and torch.isfinite(gv).all())


if __name__=='__main__':unittest.main()
