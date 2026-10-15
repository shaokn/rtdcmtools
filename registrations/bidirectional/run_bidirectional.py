"""Experimental paired B-spline with explicit two-sided inverse consistency (LPS mm).
Preserves saved unidirectional code. Supports rigid rotations by building an axis-aligned common-coordinate CT grid.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import SimpleITK as s
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(BASE/'gpu_pytorch_multilevel'))
from registration_pipeline import load_inputs, same_grid
from dose_mapping import map_optional_doses
from registration_inputs import add_explicit_arguments,validate_image_arguments,load_masks
from registration_utils import map_image, ones, write_json
from registration_utils import work_reference
from multilevel import PhysicalBspline


def normalized_query(reference, displacement, destination):
    """Reference-grid XYZ displacement mm -> destination normalized XYZ query."""
    shape=displacement.shape[:3]
    device=displacement.device
    axes=[torch.linspace(0,(n-1)*sp,n,device=device)+org
          for n,sp,org in zip(reference.GetSize()[::-1],reference.GetSpacing()[::-1],reference.GetOrigin()[::-1])]
    z,y,x=torch.meshgrid(*axes,indexing='ij')
    position=torch.stack((x,y,z),dim=-1)
    origin=torch.tensor(destination.GetOrigin(),device=device,dtype=torch.float32)
    extent=torch.tensor((np.array(destination.GetSize())-1)*destination.GetSpacing(),device=device,dtype=torch.float32)
    return (2*(position+displacement-origin)/extent-1)[None]


def sample(image,query,padding='zeros'):
    return F.grid_sample(image,query,mode='bilinear',padding_mode=padding,align_corners=True)


def ncc(target,warped,weight):
    count=weight.sum()
    a=target-(target*weight).sum()/count
    b=warped-(warped*weight).sum()/count
    return (weight*a*b).sum()/torch.sqrt((weight*a.square()).sum()*(weight*b.square()).sum()+1e-8)


def tensor(im,device,intensity=False):
    a=s.GetArrayFromImage(im).astype(np.float32)
    if intensity:a=(np.clip(a,-1000,1000)+1000)/2000
    return torch.as_tensor(a,device=device)[None,None]


def field_image(array,reference):
    im=s.GetImageFromArray(np.asarray(array,dtype=np.float64),isVector=True)
    im.CopyInformation(reference)
    return im


def summary(values):
    a=np.asarray(values)
    return dict(n=int(a.size),mean_mm=float(a.mean()),p95_mm=float(np.percentile(a,95)),max_mm=float(a.max())) if a.size else dict(n=0)


def cycle_image(first,second):
    """Both fields in the same physical frame; strict interpolation-domain coverage."""
    tr=s.DisplacementFieldTransform(s.Image(first))
    other=s.Resample(second,first,tr,s.sitkLinear,0,s.sitkVectorFloat64)
    # A float all-ones image and zero padding identify points in the second field domain.
    coverage=s.Resample(s.Cast(ones(second),s.sitkFloat32),first,tr,s.sitkLinear,0,s.sitkFloat32)
    err=np.linalg.norm(s.GetArrayFromImage(first)+s.GetArrayFromImage(other),axis=-1)
    valid=s.GetArrayFromImage(coverage)>.999
    # Resample may accept a half-voxel border. For inverse-error reporting use
    # the strict voxel-center interpolation domain, matching the training query.
    values=s.GetArrayFromImage(first)
    for component,axis in enumerate((2,1,0)):
        coordinates=(np.arange(first.GetSize()[component])*first.GetSpacing()[component]+first.GetOrigin()[component])
        shape=[1,1,1];shape[axis]=len(coordinates)
        position=coordinates.reshape(shape)+values[...,component]
        low=second.GetOrigin()[component]
        high=low+(second.GetSize()[component]-1)*second.GetSpacing()[component]
        valid &= (position>=low)&(position<=high)
    valid[[0,-1],:,:]=False;valid[:,[0,-1],:]=False;valid[:,:,[0,-1]]=False
    im=s.GetImageFromArray(err.astype(np.float32));im.CopyInformation(first)
    mask=s.GetImageFromArray(valid.astype(np.uint8));mask.CopyInformation(first)
    return im,mask


def objective(which,models,refs,images,weights,coverages,args):
    other=1-which
    d=models[which]()
    query=normalized_query(refs[which],d,refs[other])
    warped=sample(images[other],query)
    score=ncc(images[which],warped,weights[which])
    lost=(weights[which]*(1-sample(coverages[other],query)).square()).sum()/weights[which].sum()
    smooth=sum((torch.diff(d,dim=ax)/sp).square().mean() for ax,sp in enumerate(refs[which].GetSpacing()[::-1]))/3
    cycle=torch.zeros((),device=d.device)
    fraction=torch.ones((),device=d.device)
    if args.inverse_weight:
        # Subsample evaluation points only, not the image loss or either deformation field.
        # Both model coefficient sets receive cycle gradients in this SAME optimizer step.
        q=query[:,::2,::2,::2]
        dv=models[other]()
        back=sample(dv.permute(3,0,1,2)[None],q,padding='border')[0].permute(1,2,3,0)
        residual=d[::2,::2,::2]+back
        valid=(q.abs()<=1).all(dim=-1)[0].detach()
        w=weights[which][0,0,::2,::2,::2]*valid
        fraction=w.sum()/weights[which][0,0,::2,::2,::2].sum()
        cycle=(w*residual.square().sum(-1)).sum()/(3*w.sum().clamp_min(1))
    loss=1-score+args.smoothness*smooth+lost+args.inverse_weight*cycle
    return loss,dict(ncc=float(score.detach()),smoothness=float(smooth.detach()),coverage_penalty=float(lost.detach()),
        cycle_component_mse_mm2=float(cycle.detach()),cycle_valid_fraction=float(fraction.detach()))


def aligned_reference(moving,rigid):
    """Full original CT coverage in a common axis-aligned frame, including rotations."""
    import itertools
    inverse=rigid.GetInverse()
    if np.allclose(rigid.GetMatrix(),np.eye(3).ravel(),atol=1e-10):
        ref=s.Image(moving.GetSize(),s.sitkFloat32);ref.CopyInformation(moving)
        ref.SetOrigin(inverse.TransformPoint(moving.GetOrigin()))
        return ref
    corners=[inverse.TransformPoint(moving.TransformIndexToPhysicalPoint(tuple(int(v) for v in index)))
             for index in itertools.product(*[(0,n-1) for n in moving.GetSize()])]
    low=np.min(corners,axis=0);high=np.max(corners,axis=0)
    spacing=np.asarray(moving.GetSpacing())
    size=np.ceil((high-low)/spacing-1e-8).astype(int)+1
    ref=s.Image(size.tolist(),s.sitkFloat32)
    ref.SetSpacing(spacing.tolist());ref.SetOrigin(low.tolist())
    return ref


def optimize(args,fixed,moving,rigid,aligned,supports,report):
    states=None;refs_full=[fixed,aligned];images_full=[fixed,moving]
    report['levels']=[]
    for level,(factor,sigma) in enumerate(zip((4,2,1),(2.,1.,0.))):
        folder=args.output/f'level_{level}_down{factor}';folder.mkdir()
        refs=[work_reference(im,factor) for im in refs_full]
        identity=s.Transform(3,s.sitkIdentity)
        smooth=[s.SmoothingRecursiveGaussian(im,sigma) if sigma else im for im in images_full]
        image_work=[map_image(smooth[0],refs[0],identity),map_image(smooth[1],refs[1],rigid)]
        images=[tensor(im,args.device,True) for im in image_work]
        weights=[tensor(map_image(im,ref,identity,True),args.device) for im,ref in zip(supports,refs)]
        if any(w.sum().item()<100 for w in weights):raise RuntimeError('Insufficient support at this level')
        coverages=[tensor(map_image(ones(fixed),refs[0],identity,True),args.device),
                   tensor(map_image(ones(moving),refs[1],rigid,True),args.device)]
        models=[PhysicalBspline(full,ref,args.control_spacing,args.device) for full,ref in zip(refs_full,refs)]
        if states:
            for model,state in zip(models,states):model.load_state_dict(state)
        opt=torch.optim.Adam([p for model in models for p in model.parameters()],lr=args.learning_rate)
        best=float('inf');best_states=None;best_step=0
        item=dict(level=level,downsample=factor,smoothing_sigma_mm=sigma,work_sizes_xyz=[list(r.GetSize()) for r in refs])
        report['levels'].append(item)
        write_json(args.output/'report.json',report)
        start=time.monotonic()
        with (folder/'history.jsonl').open('w') as log:
            for step in range(args.iterations+1):
                opt.zero_grad(set_to_none=True)
                total=0.;rows=[]
                # Sequential directional backward avoids retaining two full-resolution graphs.
                # Both gradients are accumulated before the single shared optimizer step.
                for direction in (0,1):
                    loss,row=objective(direction,models,refs,images,weights,coverages,args)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
                    total+=float(loss.detach());rows.append(row)
                    if step<args.iterations:loss.backward()
                    del loss
                if total<best:
                    best=total;best_step=step
                    best_states=[{k:v.detach().cpu().clone() for k,v in model.state_dict().items()} for model in models]
                    best_rows=rows
                record=dict(step=step,loss=total,fbct_to_aligned_ct=rows[0],aligned_ct_to_fbct=rows[1])
                log.write(json.dumps(record)+'\n');log.flush()
                if step%20==0 or step==args.iterations:print(json.dumps(dict(level=level,**record)),flush=True)
                if step<args.iterations:
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for model in models for p in model.parameters()):raise RuntimeError('Invalid gradient')
                    opt.step()
                    with torch.no_grad():
                        for model in models:
                            for parameter in model.parameters():parameter.clamp_(-args.max_displacement,args.max_displacement)
        states=best_states
        for model,state in zip(models,states):model.load_state_dict(state)
        with torch.no_grad():
            fields=[field_image(model().cpu().numpy(),ref) for model,ref in zip(models,refs)]
        for label,field,state in zip(['fbct','aligned_ct'],fields,states):
            s.WriteImage(field,str(folder/f'residual_{label}.mha'))
            torch.save(state,folder/f'coefficients_{label}.pt')
        item.update(best_loss=best,best_step=best_step,directional_metrics=best_rows,elapsed_seconds=time.monotonic()-start)
        write_json(folder/'report.json',item)
        write_json(args.output/'report.json',report)
        del models,opt,images,weights,coverages
        if args.device=='cuda':torch.cuda.empty_cache()
    return fields


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,help=argparse.SUPPRESS)
    add_explicit_arguments(p)
    p.add_argument('--reg-root',type=Path,default=Path('.'))
    p.add_argument('--reg',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--inverse-weight',type=float,default=.1)
    p.add_argument('--control-spacing',type=float,default=30.)
    p.add_argument('--learning-rate',type=float,default=.2)
    p.add_argument('--smoothness',type=float,default=.1)
    p.add_argument('--max-displacement',type=float,default=10.)
    p.add_argument('--rois',nargs='+',default=None)
    args=p.parse_args()
    validate_image_arguments(p,args)
    if args.iterations<1 or not np.isfinite([args.inverse_weight,args.control_spacing,args.learning_rate,args.smoothness,args.max_displacement]).all() or args.inverse_weight<0 or min(args.control_spacing,args.learning_rate,args.smoothness,args.max_displacement)<=0:p.error('Invalid parameters')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('GPU unavailable')
    torch.set_num_threads(4);s.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();report=dict(status='running',method='joint_bidirectional_cubic_bspline',inverse_weight=args.inverse_weight,
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        warnings=['Experimental soft inverse consistency; not guaranteed exactly inverse/diffeomorphic; no dose mapping performed.'])
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    write_json(args.output/'report.json',report)
    try:
        moving,fixed,rigid,masks,provenance=load_inputs(args);report.update(provenance)
        if not np.allclose(moving.GetDirection(),np.eye(3).ravel(),atol=1e-6):raise ValueError('Experimental implementation requires identity original CT direction')
        aligned=aligned_reference(moving,rigid)
        identity=s.Transform(3,s.sitkIdentity)
        moving_aligned=map_image(moving,aligned,rigid)
        support_f=s.BinaryErode(map_image(ones(moving),fixed,rigid,True)&s.Cast(fixed>-700,s.sitkUInt8),[1]*3)
        support_a=s.BinaryErode(map_image(ones(fixed),aligned,identity,True)&map_image(ones(moving),aligned,rigid,True)&s.Cast(moving_aligned>-700,s.sitkUInt8),[1]*3)
        start=time.monotonic()
        u,v=optimize(args,fixed,moving,rigid,aligned,[support_f,support_a],report)
        if args.device=='cuda':torch.cuda.synchronize()
        report['registration_seconds']=time.monotonic()-start
        for name,im in [('residual_fbct_to_aligned_ct.mha',u),('residual_aligned_ct_to_fbct.mha',v),('fixed.nii.gz',fixed),('moving.nii.gz',moving),('support_fbct.nii.gz',support_f),('support_aligned_ct.nii.gz',support_a)]:s.WriteImage(im,str(args.output/name))
        report['jacobian']={}
        for label,field in [('fbct',u),('aligned_ct',v)]:
            jac=s.DisplacementFieldJacobianDeterminant(field);a=s.GetArrayFromImage(jac)[1:-1,1:-1,1:-1]
            report['jacobian'][label]=dict(min_interior=float(a.min()),nonpositive_fraction=float((a<=0).mean()))
            s.WriteImage(jac,str(args.output/f'jacobian_{label}.nii.gz'))
            if not np.isfinite(a).all() or (a<=0).any():raise RuntimeError(f'Nonpositive Jacobian: {label}; output propagation stopped')
        forward=s.CompositeTransform([rigid,s.DisplacementFieldTransform(s.Image(u))])
        reverse=s.CompositeTransform([s.DisplacementFieldTransform(s.Image(v)),rigid.GetInverse()])
        s.WriteTransform(forward,str(args.output/'pull_fixed_to_moving.h5'))
        s.WriteTransform(reverse,str(args.output/'pull_moving_to_fixed.h5'))
        s.WriteImage(map_image(moving,fixed,rigid),str(args.output/'moving_rigid_on_fixed.nii.gz'))
        s.WriteImage(map_image(moving,fixed,forward),str(args.output/'moving_registered_on_fixed.nii.gz'))
        s.WriteImage(map_image(fixed,moving,reverse),str(args.output/'fixed_registered_on_moving.nii.gz'))
        report['roundtrip']={}
        for label,first,second,support in [('fbct',u,v,support_f),('aligned_ct',v,u,support_a)]:
            err,valid=cycle_image(first,second)
            s.WriteImage(err,str(args.output/f'roundtrip_error_{label}_mm.nii.gz'))
            s.WriteImage(valid,str(args.output/f'roundtrip_valid_{label}.nii.gz'))
            ev=s.GetArrayFromImage(err);va=s.GetArrayFromImage(valid)>0
            report['roundtrip'][label]=dict(all_valid=summary(ev[va]),metric_support=summary(ev[va&(s.GetArrayFromImage(support)>0)]),valid_fraction=float(va.mean()))
            if label=='aligned_ct':
                error_ct=s.Resample(err,moving,rigid.GetInverse(),s.sitkLinear,-1,s.sitkFloat32)
                valid_ct=s.Resample(valid,moving,rigid.GetInverse(),s.sitkNearestNeighbor,0,s.sitkUInt8)
                s.WriteImage(error_ct,str(args.output/'roundtrip_error_moving_mm.nii.gz'))
                s.WriteImage(valid_ct,str(args.output/'roundtrip_valid_moving.nii.gz'))
        report['structures']=[]
        for stem,mask in masks:
            result=map_image(mask,fixed,forward,True);rigid_mask=map_image(mask,fixed,rigid,True)
            s.WriteImage(result,str(args.output/f'{stem}_bidirectional_deformed.nii.gz'))
            s.WriteImage(rigid_mask,str(args.output/f'{stem}_rigid.nii.gz'))
            # This roundtrip is diagnostic only, NOT an independent reference contour.
            back=map_image(result,moving,reverse,True)
            s.WriteImage(back,str(args.output/f'{stem}_roundtrip_on_moving.nii.gz'))
            error=s.GetArrayFromImage(s.ReadImage(str(args.output/'roundtrip_error_moving_mm.nii.gz')))
            valid=s.GetArrayFromImage(s.ReadImage(str(args.output/'roundtrip_valid_moving.nii.gz')))>0
            ma=s.GetArrayFromImage(mask)>0
            report['structures'].append(dict(name=stem,roundtrip=summary(error[ma&valid]),ct1_mask_valid_fraction=float((ma&valid).sum()/ma.sum()),output_nonempty=bool(s.GetArrayViewFromImage(result).any())))
        for stem,mask in load_masks(getattr(args,'fixed_masks',None),fixed):
            result=map_image(mask,moving,reverse,True)
            folder=args.output/'structures_on_moving';folder.mkdir(exist_ok=True)
            s.WriteImage(result,str(folder/f'{stem}_mapped.nii.gz'))
        report['dose_mapping']={
            'moving_to_fixed':map_optional_doses(report.get('moving_dose_source') if args.fixed else args.data/'plan_ct1',moving,fixed,forward,args.output/'dose_moving_to_fixed',report.get('moving_frame')),
            'fixed_to_moving':map_optional_doses(report.get('fixed_dose_source') if args.fixed else args.data/'fraction_fbct1',fixed,moving,reverse,args.output/'dose_fixed_to_moving',report.get('fixed_frame'))}
        report['status']='completed_requires_review'
    except BaseException as e:report.update(status='failed',error=repr(e));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['peak_allocated_gpu_gib']=torch.cuda.max_memory_allocated()/2**30 if args.device=='cuda' else None
        write_json(args.output/'report.json',report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
