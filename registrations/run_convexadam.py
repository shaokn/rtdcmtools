"""Optional ConvexAdam entry and working-grid helpers."""
import numpy as np
import SimpleITK as s

def work_reference(fixed, spacing_xyz, grid_sp, grid_sp_adam):
 spacing=np.asarray(spacing_xyz,dtype=float)
 if spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any():
  raise ValueError('Expected three positive finite spacings in X,Y,Z order')
 multiple=int(np.lcm(grid_sp,grid_sp_adam))
 # Cover both endpoint centers; add padding only at the upper edges.
 required=np.ceil((np.asarray(fixed.GetSize())-1)*fixed.GetSpacing()/spacing).astype(int)+1
 size=((required+multiple-1)//multiple)*multiple
 work=s.Image(size.tolist(),s.sitkFloat32)
 work.SetSpacing(spacing.tolist());work.SetOrigin(fixed.GetOrigin());work.SetDirection(fixed.GetDirection())
 return work

def displacement_to_lps(disp,work):
 # ConvexAdam channels follow numpy Z,Y,X axes, in work-grid voxel units.
 # Convert each component using its own spacing, then rotate into LPS.
 return (disp[...,::-1]*np.asarray(work.GetSpacing()))@np.asarray(work.GetDirection()).reshape(3,3).T

def main():
 from registration_pipeline import main as run
 run('convexadam')

if __name__=='__main__':main()
