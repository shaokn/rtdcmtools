"""Prefer a valid matching DICOM REG; otherwise estimate six-DOF rigid from NIfTI."""
import json
import time
from pathlib import Path
import numpy as np
import SimpleITK as s
import pydicom
from registration_utils import pull_matrix, affine_transform


def transform_matrix(transform):
    # Handles Euler centers correctly: translation parameter != homogeneous offset.
    offset=np.asarray(transform.TransformPoint((0.,0.,0.)))
    rotation=np.column_stack([np.asarray(transform.TransformPoint(tuple(float(v) for v in axis)))-offset for axis in np.eye(3)])
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(rotation),1,atol=1e-5):
        raise ValueError('Estimated initializer is not a proper rigid transformation')
    matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=offset
    return matrix


def matching_reg(root,fixed_frame,moving_frame,explicit=None):
    """Return None only for no matching REG; ambiguity/malformed files are errors."""
    if explicit:
        path=Path(explicit);ds=pydicom.dcmread(path,stop_before_pixels=True)
        return path,pull_matrix(ds,fixed_frame,moving_frame)
    root=Path(root);files=set()
    for folder in [root/'other/REG',root/'REG',root]:
        if folder.is_dir():files.update(folder.glob('*.dcm'))
    matches=[]
    for path in sorted(files):
        ds=pydicom.dcmread(path,stop_before_pixels=True)
        frames={str(e.FrameOfReferenceUID) for e in ds.get('RegistrationSequence',[])}
        if {fixed_frame,moving_frame}<=frames:matches.append((path,ds))
    if not matches:return None
    apply=[x for x in matches if 'igrtapply' in x[0].name.lower()]
    default=[x for x in matches if 'igrtdefault' in x[0].name.lower()]
    selected=apply or default or matches
    if len(selected)!=1:raise ValueError(f'Multiple matching REG files ({len(selected)}); specify --reg explicitly. No automatic fallback.')
    path,ds=selected[0]
    return path,pull_matrix(ds,fixed_frame,moving_frame)


def estimate_rigid(fixed,moving,output=None,iterations=100,sampling=.2):
    """One ordinary SimpleITK six-DOF registration call, with its built-in pyramid."""
    start=time.monotonic()
    if iterations<1 or not 0<sampling<=1:raise ValueError('Invalid rigid parameters')
    fixed=s.Clamp(s.Cast(fixed,s.sitkFloat32),lowerBound=-1000,upperBound=1000)
    moving=s.Clamp(s.Cast(moving,s.sitkFloat32),lowerBound=-1000,upperBound=1000)
    initial=s.Euler3DTransform()
    initial.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(tuple((n-1)/2 for n in fixed.GetSize())))
    reg=s.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(40)
    reg.SetMetricFixedMask(s.Cast(fixed>-700,s.sitkUInt8))
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling,42)
    reg.SetInterpolator(s.sitkLinear)
    reg.SetOptimizerAsGradientDescentLineSearch(learningRate=1.,numberOfIterations=iterations,
        convergenceMinimumValue=1e-6,convergenceWindowSize=10,
        lineSearchUpperLimit=2.,lineSearchMaximumIterations=10,maximumStepSizeInPhysicalUnits=1.)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4,2,1])
    reg.SetSmoothingSigmasPerLevel([2.,1.,0.])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial,inPlace=True)
    reg.Execute(fixed,moving)
    transform=affine_transform(transform_matrix(initial))
    report=dict(source='automatic_six_dof_rigid',metric='Mattes MI',
        final_metric=float(reg.GetMetricValue()),stop_condition=reg.GetOptimizerStopConditionDescription(),
        elapsed_seconds=time.monotonic()-start,matrix_lps=transform_matrix(transform).tolist())
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        s.WriteTransform(transform,str(output/'auto_rigid_estimated_fixed_to_moving.tfm'))
        s.WriteImage(s.Resample(moving,fixed,transform,s.sitkLinear,-1000,s.sitkFloat32),str(output/'auto_rigid_moving_on_fixed.nii.gz'))
    return transform,report


def resolve_rigid(args,fixed,moving,fixed_frame,moving_frame):
    match=matching_reg(args.reg_root,fixed_frame,moving_frame,getattr(args,'reg',None))
    if match:
        path,matrix=match
        return affine_transform(matrix),dict(source='dicom_reg',registration_dicom=str(path),matrix_lps=matrix.tolist())
    reverse=getattr(args,'rigid_estimation_direction','fbct_to_ct1')=='ct1_to_fbct'
    estimated,report=estimate_rigid(moving if reverse else fixed,fixed if reverse else moving,
        getattr(args,'output',None))
    canonical=affine_transform(transform_matrix(estimated.GetInverse() if reverse else estimated))
    report['estimation_pull_direction']='moving -> fixed' if reverse else 'fixed -> moving'
    report['canonical_pull_direction']='fixed -> moving'
    report['canonical_matrix_lps']=transform_matrix(canonical).tolist()
    if getattr(args,'output',None):
        s.WriteTransform(canonical,str(Path(args.output)/'rigid_fixed_to_moving.tfm'))
        (Path(args.output)/'auto_rigid_report.json').write_text(json.dumps(report,indent=2))
    return canonical,report

