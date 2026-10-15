import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import SimpleITK as s
import pydicom
from pydicom.dataset import FileDataset,Dataset,FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
from rigid_initialization import matching_reg,resolve_rigid,estimate_rigid,transform_matrix


def write_reg(path,translation=(3.,-2.,1.)):
    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID='1.2.840.10008.5.1.4.1.1.66.1';meta.MediaStorageSOPInstanceUID=pydicom.uid.generate_uid()
    ds=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
    ds.SOPClassUID=meta.MediaStorageSOPClassUID;ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    entries=[]
    for uid,offset in [('1.2.3',(0.,0.,0.)),('1.2.4',translation)]:
        entry=Dataset();entry.FrameOfReferenceUID=uid
        matrix=np.eye(4);matrix[:3,3]=offset
        item=Dataset();item.FrameOfReferenceTransformationMatrixType='RIGID';item.FrameOfReferenceTransformationMatrix=matrix.ravel().tolist()
        registration=Dataset();registration.MatrixSequence=[item];entry.MatrixRegistrationSequence=[registration];entries.append(entry)
    ds.RegistrationSequence=entries;ds.save_as(path,enforce_file_format=True)


class AutomaticRigidTests(unittest.TestCase):
    def test_reg_preferred_and_ambiguity_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);regdir=root/'other/REG';regdir.mkdir(parents=True)
            path=regdir/'IGRTApply_A.dcm';write_reg(path)
            a=SimpleNamespace(reg_root=root,reg=None,rigid_mode='auto')
            with patch('rigid_initialization.estimate_rigid',side_effect=AssertionError('Must not estimate when REG exists')):
                transform,report=resolve_rigid(a,None,None,'1.2.3','1.2.4')
            self.assertEqual(report['source'],'dicom_reg')
            np.testing.assert_allclose(transform.TransformPoint([0.,0.,0.]),[-3.,2.,-1.])
            write_reg(regdir/'IGRTApply_B.dcm')
            with self.assertRaisesRegex(ValueError,'Multiple matching'):resolve_rigid(a,None,None,'1.2.3','1.2.4')

    def test_no_reg_falls_back_and_reverse_direction_is_respected(self):
        with tempfile.TemporaryDirectory() as folder:
            a=SimpleNamespace(reg_root=Path(folder),reg=None,rigid_mode='auto',rigid_estimation_direction='ct1_to_fbct')
            fixed=s.Image([8]*3,s.sitkFloat32);moving=s.Image([9]*3,s.sitkFloat32)
            t=s.TranslationTransform(3,[2.,-1.,3.])
            with patch('rigid_initialization.estimate_rigid',return_value=(t,{'source':'automatic_six_dof_rigid'})) as estimate:
                canonical,report=resolve_rigid(a,fixed,moving,'1.2.3','1.2.4')
                self.assertIs(estimate.call_args.args[0],moving)
                self.assertIs(estimate.call_args.args[1],fixed)
                np.testing.assert_allclose(canonical.TransformPoint([0.,0.,0.]),[-2.,1.,-3.])

    def test_missing_explicit_reg_is_error(self):
        with tempfile.TemporaryDirectory() as folder:
            a=SimpleNamespace(reg_root=Path(folder),reg=Path(folder)/'missing.dcm',rigid_mode='auto')
            with self.assertRaises(FileNotFoundError):resolve_rigid(a,None,None,'1.2.3','1.2.4')

    def test_euler_center_is_preserved_in_affine_matrix(self):
        t=s.Euler3DTransform();t.SetCenter([40.,-20.,10.]);t.SetRotation(.05,-.08,.12);t.SetTranslation([1.,2.,3.])
        matrix=transform_matrix(t)
        for point in [[0.,0.,0.],[15.,30.,45.]]:
            np.testing.assert_allclose((matrix@[*point,1])[:3],t.TransformPoint(point),atol=1e-10)

    def test_synthetic_rotation_translation_recovery(self):
        z,y,x=np.mgrid[:48,:48,:48]
        a=-1000+900*np.exp(-((x-16)**2/35+(y-20)**2/65+(z-21)**2/70))+1400*np.exp(-((x-32)**2/45+(y-29)**2/40+(z-29)**2/60))+700*np.exp(-((x-19)**2/50+(y-33)**2/22+(z-35)**2/30))
        fixed=s.GetImageFromArray(a.astype('float32'))
        truth=s.Euler3DTransform();truth.SetCenter([23.5]*3);truth.SetRotation(.02,-.025,.035);truth.SetTranslation([2.,-1.5,1.])
        moving=s.Resample(fixed,fixed,truth.GetInverse(),s.sitkLinear,-1000,s.sitkFloat32)
        s.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
        result,report=estimate_rigid(fixed,moving,iterations=120,sampling=.5)
        for point in [[15.,20.,20.],[30.,28.,30.],[20.,33.,34.]]:
            self.assertLess(np.linalg.norm(np.array(result.TransformPoint(point))-truth.TransformPoint(point)),.1)


if __name__=='__main__':unittest.main()
