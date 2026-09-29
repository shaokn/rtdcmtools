# 转换验证说明

公开仓库不保存真实病例编号、病例级统计或 QA 数值。本文件只记录可复现的验证方法。

## 自动化测试

`tests/test_convert_dicom_to_nifti.py` 使用合成数据覆盖：

- CT 层序、HU 映射和轴顺序。
- 倾斜及反向剂量网格。
- `DoseGridScaling` 到 Gy 的单次缩放。
- 绝对 z 剂量帧位置和非均匀间距拒绝。
- ROI mask 对齐及显式裁剪保护。
- NIfTI 独立读回。

运行全部测试：

```bash
PYTHONPATH="$PWD:$PWD/viewer" .venv/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v
```

## 本地真实数据验证

真实病例验证结果必须保存在 Git 忽略的 `viewer/qa/` 或仓库外部，不得提交到公开仓库。建议至少检查：

1. CT 每层像素应用 DICOM 映射后与 NIfTI HU 一致。
2. CT 每层角点的物理坐标一致。
3. 剂量体素等于原像素乘 `DoseGridScaling`，单位为 Gy。
4. 剂量逐帧角点物理坐标一致。
5. mask 为二值，并与引用 CT 的尺寸、间距、原点和方向一致。
6. 原文件 SHA-256 未变化，输出 SHA-256 与索引一致。
7. 结构体积、DVH 和等剂量显示与 TPS 或 SlicerRT 交叉核对。

这些检查仍不等同于临床验证，也不能证明兼容所有设备和 TPS 导出格式。
