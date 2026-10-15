"""NIfTI registration, label propagation and optional dose mapping."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import SimpleITK as s
import torch
from registration_utils import affine_transform, map_image, ones, write_json
from rigid_initialization import resolve_rigid, transform_matrix
from dose_mapping import map_optional_doses
from registration_inputs import load_explicit_inputs,add_explicit_arguments,validate_image_arguments

ROOT = Path(__file__).resolve().parent


def same_grid(a, b):
    return a.GetSize() == b.GetSize() and all(np.allclose(getattr(a, k)(), getattr(b, k)(), rtol=0, atol=1e-6)
        for k in ('GetOrigin', 'GetSpacing', 'GetDirection'))


def load_inputs(args):
    if getattr(args,'fixed',None):return load_explicit_inputs(args)
    args.rois=args.rois or ['Heart','Lung_L','Lung_R','PTV4350','SpinalCord']
    moving_package = args.data / 'plan_ct1'
    fixed_package = args.data / 'fraction_fbct1'
    mi, fi = [json.loads((p/'index.json').read_text()) for p in (moving_package, fixed_package)]
    mf, ff = [i['ct']['dicom']['00200052']['Value'][0] for i in (mi, fi)]
    moving_path, fixed_path = [p/i['ct']['nifti'] for p, i in ((moving_package, mi), (fixed_package, fi))]
    moving, fixed = [s.ReadImage(str(p), s.sitkFloat32) for p in (moving_path, fixed_path)]
    if not np.allclose(fixed.GetDirection(), np.eye(3).ravel(), atol=1e-6):
        raise ValueError('This B-spline implementation requires identity fixed direction')
    metadata = json.loads((moving_package/'RS/RS1/metadata.json').read_text())
    if mi['ct']['series_uid'] not in metadata['referenced_ct_series_uids']:
        raise ValueError('Structure metadata does not reference moving CT series')
    masks = []
    for name in args.rois:
        entries = [r for r in metadata['rois'] if r['roi_name'] == name]
        if len(entries) != 1:
            raise ValueError(f'Expected exactly one ROI named {name}')
        path = moving_package/'RS/RS1/masks'/f'{name}.nii.gz'
        mask = s.ReadImage(str(path))
        a = s.GetArrayFromImage(mask)
        if not same_grid(mask, moving) or not np.isin(a, [0, 1]).all() or not a.any():
            raise ValueError(f'Invalid original mask grid/values: {path}')
        masks.append((f"{entries[0]['roi_number']:03d}_{name}", s.Cast(mask, s.sitkUInt8)))
    rigid, rigid_report = resolve_rigid(args, fixed, moving, ff, mf)
    matrix = transform_matrix(rigid)
    provenance = dict(moving_nifti=str(moving_path), fixed_nifti=str(fixed_path),
        mask_directory=str(moving_package/'RS/RS1/masks'), registration_dicom=rigid_report.get("registration_dicom"), rigid_initialization=rigid_report,
        fixed_frame=ff, moving_frame=mf, fixed_to_moving_matrix_lps=matrix.tolist(),
        source_geometry=dict(moving_size=list(moving.GetSize()), fixed_size=list(fixed.GetSize())),
        input_basis='Original NIfTI CT/FBCT and masks; matching REG when available, otherwise automatic rigid')
    return moving, fixed, rigid, masks, provenance


def convex_field(args, fixed, moving, rigid, report):
    from run_convexadam import work_reference, displacement_to_lps
    from convexadam_memory import convex_adam_pt
    spacing = args.spacing or [*fixed.GetSpacing()[:2], 2.5]
    work = work_reference(fixed, spacing, 6, 1)
    identity = s.Transform(3, s.sitkIdentity)
    def array(im, transform):
        return np.clip(s.GetArrayFromImage(map_image(im, work, transform)), -1000, 1000).astype(np.float32)
    f, m = array(fixed, identity), array(moving, rigid)
    report['work_size_xyz'] = list(work.GetSize())
    report['work_spacing_xyz_mm'] = list(work.GetSpacing())
    report['objective'] = 'MIND-SSC squared error + ConvexAdam voxel-grid diffusion; no B-spline metric mask'
    report['grid_sp'] = 6
    report['grid_sp_adam'] = 1
    report['memory_implementation'] = 'Feature-channel blocked gradient, float32; no spatial downsampling in Adam'
    write_json(args.output/'report.json', report)
    disp = convex_adam_pt(f, m, selected_niter=args.iterations, grid_sp=6, grid_sp_adam=1,
        lambda_weight=args.lambda_weight, disp_hw=args.disp_hw, selected_smooth=args.smooth,
        dtype=torch.float32, device=torch.device(args.device), verbose=True)
    vector = displacement_to_lps(disp, work)
    if not np.isfinite(vector).all():
        raise RuntimeError('Nonfinite ConvexAdam displacement')
    field = s.GetImageFromArray(np.ascontiguousarray(vector), isVector=True)
    field.CopyInformation(work)
    return s.Resample(field, fixed, identity, s.sitkLinear, 0, s.sitkVectorFloat64)


def main(method=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['bspline', 'convexadam'], default=method or 'bspline')
    p.add_argument('--data',type=Path,help=argparse.SUPPRESS)
    add_explicit_arguments(p)
    p.add_argument('--reg-root', type=Path, default=Path('.'))
    p.add_argument('--reg', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--rois', nargs='+', default=None)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--iterations', type=int)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--control-spacing', type=float, default=30.)
    p.add_argument('--learning-rate', type=float, default=.2)
    p.add_argument('--smoothness', type=float, default=.1)
    p.add_argument('--max-displacement', type=float, default=10.)
    p.add_argument('--spacing', nargs=3, type=float)
    p.add_argument('--lambda-weight', type=float, default=5.)
    p.add_argument('--disp-hw', type=int, default=2)
    p.add_argument('--smooth', type=int, default=3)
    a = p.parse_args()
    validate_image_arguments(p,a)
    a.iterations = (100 if a.method=='bspline' else 80) if a.iterations is None else a.iterations
    a.output = a.output or Path('output')/('bspline_forward' if a.method=='bspline' else 'convexadam')
    values=[a.control_spacing,a.learning_rate,a.smoothness,a.max_displacement,a.lambda_weight]
    if min(a.iterations,a.threads,a.disp_hw)<1 or not np.isfinite(values).all() or min(values)<=0 or a.smooth<0 or (a.smooth and a.smooth%2==0):
        p.error('Positive finite parameters and zero/odd smoothing required')
    if a.spacing and (not np.isfinite(a.spacing).all() or min(a.spacing)<=0):p.error('Invalid spacing')
    if a.device=='cuda' and not torch.cuda.is_available():p.error('GPU unavailable; no automatic fallback')
    torch.set_num_threads(a.threads);s.ProcessObject.SetGlobalDefaultNumberOfThreads(a.threads)
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report=dict(status='running', method=a.method, device=a.device, settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        warnings=['Grid spacing and positive Jacobian do not establish anatomical accuracy. Original Z=5 mm; interpolation adds no acquired detail.'])
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
    write_json(a.output/'report.json', report)
    try:
        moving,fixed,rigid,masks,provenance=load_inputs(a)
        report.update(provenance)
        support=s.BinaryErode(map_image(ones(moving),fixed,rigid,True) & s.Cast(fixed>-700,s.sitkUInt8),[1,1,1])
        if not s.GetArrayViewFromImage(support).any():raise ValueError('Empty metric support')
        report['support_generation']='erode(resample(ones(original CT), fixed, rigid) AND (fixed > -700 HU), radius 1 voxel); regenerated every run'
        report['support_used_for_objective']=a.method=='bspline'
        for name,im in [('fixed.nii.gz',fixed),('moving.nii.gz',moving),('moving_rigid_on_fixed.nii.gz',map_image(moving,fixed,rigid)),('metric_support.nii.gz',support)]:s.WriteImage(im,str(a.output/name))
        s.WriteTransform(rigid,str(a.output/'rigid_fixed_to_moving.tfm'))
        write_json(a.output/'report.json',report)
        t=time.monotonic()
        if a.method=='bspline':
            sys.path.insert(0,str(ROOT/'gpu_pytorch_multilevel'))
            from multilevel import optimize
            field=optimize(a,fixed,moving,rigid,support,report)
        else:field=convex_field(a,fixed,moving,rigid,report)
        if a.device=='cuda':torch.cuda.synchronize()
        report['registration_seconds']=time.monotonic()-t
        if not same_grid(field,fixed) or not np.isfinite(s.GetArrayViewFromImage(field)).all():raise RuntimeError('Invalid residual grid/data')
        s.WriteImage(field,str(a.output/'residual_lps_mm.mha'))
        residual=s.DisplacementFieldTransform(s.Image(field))
        total=s.CompositeTransform([rigid,residual])
        jac=s.DisplacementFieldJacobianDeterminant(field);j=s.GetArrayFromImage(jac)[1:-1,1:-1,1:-1]
        report.update(jacobian_min_interior=float(j.min()), nonpositive_jacobian_fraction=float((j<=0).mean()))
        s.WriteImage(jac,str(a.output/'jacobian.nii.gz'))
        if not np.isfinite(j).all() or (j<=0).any():raise RuntimeError('Folding field: structure export stopped')
        s.WriteTransform(total,str(a.output/'total_fixed_to_moving.h5'))
        s.WriteImage(map_image(moving,fixed,total),str(a.output/'moving_registered_on_fixed.nii.gz'))
        report['structures']=[]
        dest=a.output/'structures_on_fixed';dest.mkdir()
        for stem,mask in masks:
            rigid_mask=map_image(mask,fixed,rigid,True)
            warped=map_image(mask,fixed,total,True)
            b=s.GetArrayFromImage(warped)
            if not b.any() or not np.isin(b,[0,1]).all():raise RuntimeError(f'Invalid output mask: {stem}')
            for folder in (a.output,dest):
                s.WriteImage(rigid_mask,str(folder/f'{stem}_rigid.nii.gz'))
                s.WriteImage(warped,str(folder/f'{stem}_{a.method}_deformed.nii.gz'))
            report['structures'].append(dict(name=stem,file=f'{stem}_{a.method}_deformed.nii.gz',volume_cc=float(b.sum()*np.prod(fixed.GetSpacing())/1000),changed_voxels_vs_rigid=int(np.count_nonzero(b!=s.GetArrayViewFromImage(rigid_mask))),touches_boundary=any(bool(np.take(b,e,axis=ax).any()) for ax in range(3) for e in (0,-1))))
        report['mask_sampling']='Original CT NIfTI mask sampled once with rigid(residual(x)); no previously rigid-resampled mask input'
        dose_source=report.get('moving_dose_source') if a.fixed else a.data/'plan_ct1'
        report['dose_mapping']=map_optional_doses(dose_source,moving,fixed,total,a.output/'dose_moving_to_fixed',report.get('moving_frame'))
        report['status']='completed_requires_review'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['peak_allocated_gpu_gib']=torch.cuda.max_memory_allocated()/2**30 if a.device=='cuda' else None
        write_json(a.output/'report.json',report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
