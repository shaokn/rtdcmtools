"""Optional scalar-dose mapping inside the registration run; no separate CLI."""
import json
from pathlib import Path
import numpy as np
import SimpleITK as s


def map_optional_doses(source_package,source_ct,target,pull,output,expected_frame=None):
    """Map RD/*.nii[.gz] with the SAME target->source pull as image/labels.
    Preserve source scalar units; do not reapply DoseGridScaling or multiply Jacobian.
    """
    if source_package is None:return dict(status='skipped_no_dose_nifti',files=[])
    source=Path(source_package)
    package=source.parent if source.is_file() else source
    folder=package/'RD' if (package/'RD').is_dir() else package
    files=[source] if source.is_file() and (source.name.endswith('.nii.gz') or source.suffix=='.nii') else sorted([*folder.glob('*.nii.gz'),*folder.glob('*.nii')])
    if not files:return dict(status='skipped_no_dose_nifti',source_package=str(package),files=[])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    index_path=package/'index.json'
    index=json.loads(index_path.read_text()) if index_path.is_file() else {}
    expected_frame=expected_frame or index.get('ct',{}).get('dicom',{}).get('00200052',{}).get('Value',[None])[0]
    # Coverage is acquisition support, not an anatomical/registration accuracy mask.
    ct_ones=s.Image(source_ct.GetSize(),s.sitkUInt8)+1;ct_ones.CopyInformation(source_ct)
    ct_coverage=s.Resample(ct_ones,target,pull,s.sitkNearestNeighbor,0,s.sitkUInt8)
    rows=[]
    for path in files:
        stem=path.name[:-7] if path.name.endswith('.nii.gz') else path.name[:-4]
        sidecar=path.with_name(stem+'.json')
        metadata=json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        frame=metadata.get('dicom',{}).get('00200052',{}).get('Value',[None])[0]
        if expected_frame and frame and frame!=expected_frame:raise ValueError(f'Dose/CT FrameOfReferenceUID mismatch: {path}')
        if metadata.get('dose_grid_scaling_applied') is False:raise ValueError(f'Dose NIfTI scaling not applied; convert to dose units first: {path}')
        dose=s.ReadImage(str(path),s.sitkFloat32)
        if dose.GetDimension()!=3 or dose.GetNumberOfComponentsPerPixel()!=1:raise ValueError(f'Expected scalar 3D dose: {path}')
        values=s.GetArrayViewFromImage(dose)
        if not np.isfinite(values).all() or (values<0).any():raise ValueError(f'Nonfinite/negative dose values: {path}')
        dose_ones=s.Image(dose.GetSize(),s.sitkUInt8)+1;dose_ones.CopyInformation(dose)
        coverage=s.Resample(dose_ones,target,pull,s.sitkNearestNeighbor,0,s.sitkUInt8)&ct_coverage
        mapped=s.Resample(dose,target,pull,s.sitkLinear,0,s.sitkFloat32)
        mapped=s.Mask(mapped,coverage,outsideValue=0)
        dose_name=stem+'_mapped.nii.gz';coverage_name=stem+'_coverage.nii.gz'
        s.WriteImage(mapped,str(output/dose_name));s.WriteImage(coverage,str(output/coverage_name))
        valid=s.GetArrayViewFromImage(coverage)>0
        after=s.GetArrayViewFromImage(mapped)
        row=dict(source=str(path),dose_file=dose_name,coverage_file=coverage_name,
            dose_units=metadata.get('dose_units','unknown'),dose_summation_type=metadata.get('dose_summation_type','unknown'),
            dose_grid_scaling_reapplied=False,source_frame=frame,
            source_grid_xyz=list(dose.GetSize()),source_spacing_xyz_mm=list(dose.GetSpacing()),
            target_grid_xyz=list(target.GetSize()),valid_fraction=float(valid.mean()),
            source_max=float(values.max()),mapped_max_in_coverage=float(after[valid].max()) if valid.any() else None,
            interpolation='linear',operation='scalar dose pull mapping; no dose recalculation or accumulation',
            coverage_definition='Intersection of source dose grid and source CT acquisition support. Outside stored as 0 with coverage=0; not a measured zero dose.')
        if row['dose_units']=='unknown':row['warning']='No dose-unit metadata found; values preserved without conversion.'
        (output/(stem+'_mapping.json')).write_text(json.dumps(row,indent=2))
        rows.append(row)
    result=dict(status='mapped',source_package=str(package),output=str(output),files=rows,
        note='PLAN and BEAM files mapped independently, never summed. Registration/dosimetric accuracy not established.')
    (output/'report.json').write_text(json.dumps(result,indent=2))
    return result
