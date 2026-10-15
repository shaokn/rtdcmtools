"""Three image resolutions sharing one physical cubic B-spline control lattice."""
import json
import time
from types import SimpleNamespace
import numpy as np
import SimpleITK as sitk
import torch
from airlab.utils.kernelFunction import bspline_kernel_1d
from registration_utils import work_reference, map_image, ones, write_json


class PhysicalBspline(torch.nn.Module):
    """Identical coefficient lattice at every image resolution; coefficients in mm."""
    def __init__(self, fixed, work, control_spacing, device):
        super().__init__()
        full_shape = tuple(fixed.GetSize()[::-1])
        self.strides = [max(2, round(control_spacing / s)) for s in fixed.GetSpacing()[::-1]]
        strides = np.asarray(self.strides)
        count = np.ceil(np.asarray(full_shape) / strides).astype(int)
        difference = strides * count - (strides - 1) - np.asarray(full_shape)
        crop = (np.floor(np.abs(difference)/2) * np.sign(difference)
                + np.remainder(difference, 2) * np.sign(difference)).astype(int)
        self.coefficients = torch.nn.Parameter(torch.zeros(
            (1, 3, *(count + 2).tolist()), dtype=torch.float32, device=device))
        self.bases = []
        for axis, (n, m, stride) in enumerate(zip(full_shape, work.GetSize()[::-1], self.strides)):
            kernel = bspline_kernel_1d(stride, order=3, asTensor=True)
            # Work-reference preserves endpoints. Fractional full-grid coordinates
            # sample the same physical B-spline field, without resizing coefficients.
            position = torch.linspace(0, n-1, m) + stride + int(crop[axis])
            control = torch.arange(self.coefficients.shape[axis+2])
            index = position[:, None] + int((len(kernel)-1)/2) - control[None, :] * stride
            lo = index.floor().long()
            fraction = index - lo
            def sample(i):
                return torch.where((i >= 0) & (i < len(kernel)), kernel[i.clamp(0, len(kernel)-1)], 0)
            basis = sample(lo) * (1-fraction) + sample(lo+1) * fraction
            self.bases.append(basis.to(device=device, dtype=torch.float32))

    def forward(self):
        flow = self.coefficients
        for axis, basis in enumerate(self.bases, 2):
            flow = (flow.movedim(axis, -1) @ basis.T).movedim(-1, axis)
        return flow[0].permute(1, 2, 3, 0)


def _optimize_level(args, fixed, moving, rigid, support, report):
    # Lazy imports allow header-only inspection without AirLab/PyTorch.
    import torch
    import torch.nn.functional as F

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable. No automatic CPU fallback; use r96x.')
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    report.update(torch_version=str(torch.__version__), hip_version=torch.version.hip,
                  device=str(device), device_name=torch.cuda.get_device_name(device)
                  if device.type == 'cuda' else 'CPU')
    work = work_reference(fixed, args.downsample)
    identity = sitk.Transform(3, sitk.sitkIdentity)
    # Anti-alias before sampling at lower resolution. Original CT remains untouched.
    sigma = args.smoothing_sigma
    sf = sitk.SmoothingRecursiveGaussian(fixed, sigma) if sigma else fixed
    sm = sitk.SmoothingRecursiveGaussian(moving, sigma) if sigma else moving
    fixed_work = map_image(sf, work, identity)
    moving_work = map_image(sm, work, rigid)
    support_work = map_image(support, work, identity, mask=True)
    coverage_work = map_image(ones(moving), work, rigid, mask=True)

    def tensor(image, intensity=False):
        a = sitk.GetArrayFromImage(image).astype(np.float32)
        if intensity:
            a = (np.clip(a, -1000, 1000) + 1000) / 2000
        return torch.as_tensor(a, device=device)[None, None]

    target, source = tensor(fixed_work, True), tensor(moving_work, True)
    weight, coverage = tensor(support_work), tensor(coverage_work)
    if weight.sum().item() < 100:
        raise ValueError('Insufficient working-grid metric support')
    shape = tuple(target.shape[2:])  # Z,Y,X array axes; vector components are X,Y,Z.
    z, y, x = torch.meshgrid(
        *[torch.linspace(-1, 1, n, device=device) for n in shape], indexing='ij')
    grid = torch.stack((x, y, z), dim=-1)[None]
    extent = torch.tensor((np.asarray(work.GetSize()) - 1) * work.GetSpacing(),
                          dtype=torch.float32, device=device)
    model = PhysicalBspline(fixed, work, args.control_spacing, device)
    sigma_vox = model.strides
    if args.initial_state is not None:
        model.load_state_dict(args.initial_state)
    # Its linear B-spline basis is unit-independent: here coefficients are mm,
    # then explicitly converted to grid_sample normalized coordinates below.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    report.update(work_size_xyz=list(work.GetSize()), work_spacing_xyz=list(work.GetSpacing()),
                  control_stride_full_grid_zyx=sigma_vox, objective='masked global NCC + diffusion + coverage penalty',
                  transform='Shared physical lattice; separable AirLab cubic sampled basis, coefficients in axis-mm')
    write_json(args.output / 'report.json', report)

    def objective():
        displacement_mm = model()
        if tuple(displacement_mm.shape) != shape + (3,):
            raise RuntimeError('Unexpected AirLab displacement layout')
        query = grid + (2 * displacement_mm / extent)[None]
        warped = F.grid_sample(source, query, mode='bilinear',
                               padding_mode='zeros', align_corners=True)
        covered = F.grid_sample(coverage, query, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        # Keep weights fixed: do not reward the optimizer for discarding hard regions.
        count = weight.sum()
        a = target - (target * weight).sum() / count
        b = warped - (warped * weight).sum() / count
        va, vb = (weight * a.square()).sum(), (weight * b.square()).sum()
        ncc = (weight * a * b).sum() / torch.sqrt(va * vb + 1e-8)
        smooth = sum((torch.diff(displacement_mm, dim=axis) / spacing).square().mean()
                     for axis, spacing in enumerate(work.GetSpacing()[::-1])) / 3
        lost = (weight * (1 - covered).square()).sum() / count
        loss = 1 - ncc + args.smoothness * smooth + lost
        return loss, ncc, smooth, lost, displacement_mm

    initial_ncc = None
    best_loss, best_state, best_step = float('inf'), None, None
    with (args.output / 'history.jsonl').open('w', encoding='utf-8') as log:
        # Include the final post-update state and keep the best observed objective.
        for step in range(args.iterations + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, ncc, smooth, lost, displacement = objective()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite objective')
            if initial_ncc is None:
                initial_ncc = float(ncc.detach())
            value = float(loss.detach())
            if value < best_loss:
                best_loss, best_step = value, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError('Missing/nonfinite gradient')
            row = dict(level=args.level, downsample=args.downsample, step=step, loss=value, ncc=float(ncc.detach()),
                       smoothness=float(smooth.detach()), coverage_penalty=float(lost.detach()),
                       gradient_absmax=max(float(g.abs().max()) for g in gradients),
                       displacement_max_mm=float(displacement.detach().norm(dim=-1).max()))
            log.write(json.dumps(row) + '\n')
            log.flush()
            if step % 10 == 0 or step == args.iterations:
                print(json.dumps(row), flush=True)
            if step < args.iterations:
                optimizer.step()
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.clamp_(-args.max_displacement, args.max_displacement)
    model.load_state_dict(best_state)
    with torch.no_grad():
        _, final_ncc, _, _, displacement = objective()
        array = displacement.cpu().numpy().astype(np.float64)
    report.update(initial_ncc=initial_ncc, final_ncc=float(final_ncc), best_step=best_step,
                  best_loss=best_loss, gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30
                  if device.type == 'cuda' else None)
    # Local image-axis mm -> physical LPS mm. Keep numpy spatial order Z,Y,X.
    direction = np.asarray(work.GetDirection()).reshape(3, 3)
    array = array @ direction.T
    field = sitk.GetImageFromArray(array, isVector=True)
    field.CopyInformation(work)
    sitk.WriteImage(field, str(args.output / 'residual_work_lps_mm.mha'))
    return field, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}



def optimize(args, fixed, moving, rigid, support, report):
    state = None
    levels = []
    report['levels'] = levels
    root = args.output
    for level, (factor, sigma) in enumerate(zip((4, 2, 1), (2., 1., 0.))):
        output = root / f'level_{level}_down{factor}'
        output.mkdir()
        settings = SimpleNamespace(**{**vars(args), 'output': output, 'downsample': factor,
                                     'smoothing_sigma': sigma, 'initial_state': state, 'level': level})
        item = dict(level=level, downsample=factor, smoothing_sigma_mm=sigma)
        levels.append(item)
        write_json(root / 'report.json', report)
        started = time.monotonic()
        field, state = _optimize_level(settings, fixed, moving, rigid, support, item)
        item['elapsed_seconds'] = time.monotonic() - started
        torch.save(state, output / 'control_coefficients_mm.pt')
        write_json(output / 'report.json', item)
        write_json(root / 'report.json', report)
    report.update(device=args.device, torch_version=str(torch.__version__),
                  initial_ncc=levels[-1]['initial_ncc'], final_ncc=levels[-1]['final_ncc'])
    # Last level is exactly the original fixed grid. No final displacement upsampling.
    return field
