"""Explicit fixed/moving NIfTI inputs; filenames do not determine direction."""
import json
from pathlib import Path
import numpy as np
import SimpleITK as s
from rigid_initialization import resolve_rigid,transform_matrix


def stem(path):
    return path.name[:-7] if path.name.endswith('.nii.gz') else path.stem


def image_metadata(path):
    path=Path(path).resolve()
    for package in [path.parent,path.parent.parent]:
        index=package/'index.json'
        if index.is_file():
            data=json.loads(index.read_text());ct=data.get('ct',{})
            if ct.get('nifti') and (package/ct['nifti']).resolve()==path:
                return ct,package
    sidecar=path.with_name(stem(path)+'.json')
    return (json.loads(sidecar.read_text()) if sidecar.is_file() else {}),None


def frame_uid(metadata):
    return metadata.get('frame_of_reference_uid') or metadata.get('dicom',{}).get('00200052',{}).get('Value',[None])[0]


def load_masks(folder,reference,rois=None):
    if folder is None:return []
    folder=Path(folder)
    if not folder.is_dir():raise FileNotFoundError(folder)
    files=sorted([*folder.glob('*.nii.gz'),*folder.glob('*.nii')])
    selected=[];found=set()
    for path in files:
        name=stem(path)
        short=name.split('_',1)[1] if '_' in name and name.split('_',1)[0].isdigit() else name
        if rois and name not in rois and short not in rois:continue
        mask=s.ReadImage(str(path));a=s.GetArrayFromImage(mask)
        same=mask.GetSize()==reference.GetSize() and all(np.allclose(getattr(mask,k)(),getattr(reference,k)(),rtol=0,atol=1e-6) for k in ['GetOrigin','GetSpacing','GetDirection'])
        if not same or not np.isin(a,[0,1]).all() or not a.any():raise ValueError(f'Invalid binary mask/grid: {path}')
        selected.append((name,s.Cast(mask,s.sitkUInt8)));found.update([name,short])
    if rois and set(rois)-found:raise ValueError(f'Missing masks: {sorted(set(rois)-found)}')
    return selected


def load_explicit_inputs(args):
    moving_path=Path(args.moving);fixed_path=Path(args.fixed)
    moving,fixed=[s.ReadImage(str(p),s.sitkFloat32) for p in [moving_path,fixed_path]]
    if moving.GetDimension()!=3 or fixed.GetDimension()!=3:raise ValueError('Expected two 3D images')
    if not np.allclose(fixed.GetDirection(),np.eye(3).ravel(),atol=1e-6):raise ValueError('Current implementation requires identity fixed image direction')
    mm,mp=image_metadata(moving_path);fm,fp=image_metadata(fixed_path)
    mf=getattr(args,'moving_frame',None) or frame_uid(mm)
    ff=getattr(args,'fixed_frame',None) or frame_uid(fm)
    if args.reg and not (ff and mf):raise ValueError('REG requires Frame UIDs: retain image index.json or provide --fixed-frame and --moving-frame')
    moving_masks=getattr(args,'moving_masks',None)
    fixed_masks=getattr(args,'fixed_masks',None)
    if moving_masks is None and mp and (mp/'RS/RS1/masks').is_dir():moving_masks=mp/'RS/RS1/masks'
    if fixed_masks is None and fp and (fp/'RS/RS1/masks').is_dir():fixed_masks=fp/'RS/RS1/masks'
    masks=load_masks(moving_masks,moving,getattr(args,'rois',None))
    args.fixed_masks=fixed_masks
    rigid,rr=resolve_rigid(args,fixed,moving,ff,mf)
    moving_dose=getattr(args,'moving_dose',None) or (mp/'RD' if mp else None)
    fixed_dose=getattr(args,'fixed_dose',None) or (fp/'RD' if fp else None)
    provenance=dict(moving_nifti=str(moving_path),fixed_nifti=str(fixed_path),mask_directory=str(moving_masks) if moving_masks else None,
        fixed_mask_directory=str(fixed_masks) if fixed_masks else None,
        moving_dose_source=str(moving_dose) if moving_dose else None,fixed_dose_source=str(fixed_dose) if fixed_dose else None,
        fixed_frame=ff,moving_frame=mf,registration_dicom=rr.get('registration_dicom'),rigid_initialization=rr,
        fixed_to_moving_matrix_lps=transform_matrix(rigid).tolist(),input_basis='Explicit fixed/moving NIfTI paths')
    return moving,fixed,rigid,masks,provenance


def add_explicit_arguments(parser):
    parser.add_argument('--fixed',type=Path,help='Reference image; results use this grid')
    parser.add_argument('--moving',type=Path,help='Image to register onto fixed')
    parser.add_argument('--moving-masks',type=Path,help='Optional binary-mask directory on moving grid')
    parser.add_argument('--fixed-masks',type=Path,help='Optional fixed-grid masks for bidirectional propagation')
    parser.add_argument('--moving-dose',type=Path,help='Optional moving dose NIfTI or dose directory')
    parser.add_argument('--fixed-dose',type=Path,help='Optional fixed dose for bidirectional mapping')
    parser.add_argument('--fixed-frame',help='Fixed FrameOfReferenceUID when unavailable in image metadata')
    parser.add_argument('--moving-frame',help='Moving FrameOfReferenceUID when unavailable in image metadata')


def validate_image_arguments(parser,args):
    if bool(args.fixed)!=bool(args.moving):parser.error('--fixed and --moving must be provided together')
    if not args.fixed and not args.data:parser.error('Provide --fixed and --moving')
