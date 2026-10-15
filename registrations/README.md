# B-spline 形变配准

三级 B-spline 优化，支持结构传播和可选剂量映射。

## 安装

先安装适合本机的 PyTorch，再执行：

```bash
python -m pip install -r requirements.txt
```

## 运行

在本目录执行。`fixed` 是参考图像，结果与它使用相同网格；`moving` 是待配准图像。文件名任意，反向运行只需交换两者及对应的结构、剂量输入。

```bash
# 独立单向优化：moving → fixed
python run_gpu_bspline.py --fixed fixed_ct.nii.gz --moving moving_ct.nii.gz --output output/single

# 双向联合优化：加入逆一致性约束
python bidirectional/run_bidirectional.py --fixed fixed_ct.nii.gz --moving moving_ct.nii.gz --output output/bidirectional
```

可选参数：

- `--moving-masks DIR`：moving 上的二值结构目录。
- `--moving-dose FILE`：moving 上的剂量 NIfTI；也可指定剂量目录。
- `--fixed-masks DIR`、`--fixed-dose FILE`：双向模式中另一侧的结构和剂量。
- `--reg FILE.dcm`：刚性 REG；没有匹配 REG 时自动估计刚性初始化。使用 REG 需保留影像坐标系元数据，或指定 `--fixed-frame`、`--moving-frame`。
- `--device cpu`：使用 CPU；默认 `cuda`，兼容 CUDA／ROCm。

已有转换数据旁的 `index.json`、结构目录和 RD 剂量目录可自动识别。没有剂量就跳过；输出目录须不存在。其他参数见 `--help`。

## 输出

配准图像、传播结构、变换、Jacobian 检查和耗时报告。结构采用最近邻插值，剂量采用线性插值，剂量单位保持不变、不自动累加。

剂量同时输出覆盖 mask，覆盖外的零值表示无有效数据。仅供科研，配准和剂量映射精度需另行验证。
