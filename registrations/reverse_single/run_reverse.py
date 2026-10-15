"""Independent FBCT -> CT1 single-direction three-level B-spline; no cycle loss."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import SimpleITK as s
import torch
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'gpu_pytorch_multilevel'))
from registration_pipeline import load_inputs,same_grid
from dose_mapping import map_optional_doses
from registration_utils import map_image,ones,write_json
from multilevel import optimize


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data',type=Path,required=True)
 p.add_argument('--reg-root',type=Path,default=Path('.'))
 p.add_argument('--reg',type=Path)
 p.add_argument('--output',type=Path,default=Path('output/bspline_reverse'))
 p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
 p.add_argument('--iterations',type=int,default=100)
 p.add_argument('--control-spacing',type=float,default=30.)
 p.add_argument('--learning-rate',type=float,default=.2)
 p.add_argument('--smoothness',type=float,default=.1)
 p.add_argument('--max-displacement',type=float,default=10.)
 p.add_argument('--rois',nargs='+',default=['Heart','Lung_L','Lung_R','PTV4350','SpinalCord'])
 a=p.parse_args();a.rigid_estimation_direction='ct1_to_fbct'
 values=[a.control_spacing,a.learning_rate,a.smoothness,a.max_displacement]
 if a.iterations<1 or not np.isfinite(values).all() or min(values)<=0:p.error('Invalid parameters')
 if a.device=='cuda' and not torch.cuda.is_available():p.error('GPU unavailable')
 torch.set_num_threads(4);s.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
 a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 report=dict(status='running',method='independent_reverse_single_bspline',logical_direction='FBCT image to CT1 grid',
     pull_direction='CT1 LPS position -> FBCT LPS sampling position',inverse_consistency_weight=0,
     settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
 write_json(a.output/'report.json',report)
 try:
  ct1,fbct,forward_rigid,masks,provenance=load_inputs(a)
  report['input_provenance']=provenance
  rigid=forward_rigid.GetInverse()
  support=s.BinaryErode(map_image(ones(fbct),ct1,rigid,True)&s.Cast(ct1>-700,s.sitkUInt8),[1]*3)
  if not s.GetArrayViewFromImage(support).any():raise ValueError('Empty reverse support')
  for name,im in [('ct1_original.nii.gz',ct1),('fixed_fbct.nii.gz',fbct),('fbct_rigid_on_ct1.nii.gz',map_image(fbct,ct1,rigid)),('metric_support_ct1.nii.gz',support)]:s.WriteImage(im,str(a.output/name))
  s.WriteTransform(rigid,str(a.output/'rigid_ct1_to_fbct.tfm'))
  t=time.monotonic();field=optimize(a,ct1,fbct,rigid,support,report)
  if a.device=='cuda':torch.cuda.synchronize()
  report['registration_seconds']=time.monotonic()-t
  if not same_grid(field,ct1) or not np.isfinite(s.GetArrayViewFromImage(field)).all():raise RuntimeError('Invalid field')
  s.WriteImage(field,str(a.output/'residual_on_ct1_lps_mm.mha'))
  total=s.CompositeTransform([rigid,s.DisplacementFieldTransform(s.Image(field))])
  totalfield=s.TransformToDisplacementField(total,s.sitkVectorFloat64,ct1.GetSize(),ct1.GetOrigin(),ct1.GetSpacing(),ct1.GetDirection())
  s.WriteImage(totalfield,str(a.output/'total_pull_ct1_to_fbct_lps_mm.mha'))
  jac=s.DisplacementFieldJacobianDeterminant(totalfield);j=s.GetArrayFromImage(jac)[1:-1,1:-1,1:-1]
  report.update(jacobian_min_interior=float(j.min()),nonpositive_jacobian_fraction=float((j<=0).mean()))
  s.WriteImage(jac,str(a.output/'jacobian_ct1.nii.gz'))
  if not np.isfinite(j).all() or (j<=0).any():raise RuntimeError('Invalid/folding reverse transform')
  s.WriteTransform(total,str(a.output/'pull_ct1_to_fbct.h5'))
  warped=map_image(fbct,ct1,total);s.WriteImage(warped,str(a.output/'fbct_registered_on_ct1.nii.gz'))
  aa=np.clip(s.GetArrayFromImage(ct1),-1000,1000);bb=np.clip(s.GetArrayFromImage(warped),-1000,1000);w=s.GetArrayFromImage(support)>0
  report['reverse_image_ncc_on_support']=float(np.corrcoef(aa[w],bb[w])[0,1])
  report['dose_mapping']=map_optional_doses(a.data/'fraction_fbct1',fbct,ct1,total,a.output/'dose_fbct_to_ct1')
  report['status']='completed_requires_review'
 except BaseException as e:report.update(status='failed',error=repr(e));raise
 finally:
  report['elapsed_seconds']=time.monotonic()-start
  report['peak_allocated_gpu_gib']=torch.cuda.max_memory_allocated()/2**30 if a.device=='cuda' else None
  write_json(a.output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
