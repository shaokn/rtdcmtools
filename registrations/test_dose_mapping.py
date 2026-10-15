import json,tempfile,unittest
from pathlib import Path
import numpy as np
import SimpleITK as s
from dose_mapping import map_optional_doses


class DoseMappingTests(unittest.TestCase):
    def fixture(self,root):
        package=root/'input';(package/'RD').mkdir(parents=True)
        (package/'index.json').write_text(json.dumps({'ct':{'dicom':{'00200052':{'Value':['1.2.3']}}}}))
        dose=s.GetImageFromArray(np.broadcast_to(np.arange(4,dtype='float32')*2,(3,3,4)).copy());dose.SetSpacing([2.]*3)
        s.WriteImage(dose,str(package/'RD/PLAN.nii.gz'))
        metadata={'dose_units':'Gy','dose_summation_type':'PLAN','source_dose_grid_scaling':.01,'dose_grid_scaling_applied':True,'dicom':{'00200052':{'Value':['1.2.3']}}}
        (package/'RD/PLAN.json').write_text(json.dumps(metadata))
        return package,s.Image([7,5,5],s.sitkFloat32),s.Image([9,5,5],s.sitkFloat32)

    def test_optional_missing_dose_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);im=s.Image([4]*3,s.sitkFloat32)
            report=map_optional_doses(root/'missing',im,im,s.Transform(3,s.sitkIdentity),root/'output')
            self.assertEqual(report['status'],'skipped_no_dose_nifti');self.assertFalse((root/'output').exists())

    def test_linear_physical_sampling_different_grid_and_no_double_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package,source,target=self.fixture(root)
            report=map_optional_doses(package,source,target,s.TranslationTransform(3,[.5,0,0]),root/'output')
            a=s.GetArrayFromImage(s.ReadImage(str(root/'output/PLAN_mapped.nii.gz')))
            valid=s.GetArrayFromImage(s.ReadImage(str(root/'output/PLAN_coverage.nii.gz')))
            self.assertAlmostEqual(float(a[2,2,1]),1.5,places=5)
            self.assertEqual(int(valid[2,2,8]),0);self.assertEqual(float(a[2,2,8]),0)
            self.assertEqual(report['files'][0]['dose_units'],'Gy')
            self.assertFalse(report['files'][0]['dose_grid_scaling_reapplied'])

    def test_source_ct_coverage_is_required_even_if_dose_grid_extends(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package,_,target=self.fixture(root)
            source=s.Image([3,5,5],s.sitkFloat32)
            map_optional_doses(package,source,target,s.Transform(3,s.sitkIdentity),root/'output')
            valid=s.GetArrayFromImage(s.ReadImage(str(root/'output/PLAN_coverage.nii.gz')))
            a=s.GetArrayFromImage(s.ReadImage(str(root/'output/PLAN_mapped.nii.gz')))
            self.assertEqual(int(valid[2,2,4]),0);self.assertEqual(float(a[2,2,4]),0)

    def test_mismatched_dose_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package,source,target=self.fixture(root)
            p=package/'RD/PLAN.json';m=json.loads(p.read_text());m['dicom']['00200052']['Value']=['1.2.4'];p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'FrameOfReferenceUID mismatch'):
                map_optional_doses(package,source,target,s.Transform(3,s.sitkIdentity),root/'output')


if __name__=='__main__':unittest.main()
