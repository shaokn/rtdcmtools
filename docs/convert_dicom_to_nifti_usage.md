# 整理后的 DICOM 转 NIfTI

脚本：`convert_dicom_to_nifti.py`。独立命令行工具，不需要启动网页。
输入为 `organize_dicom.py` 的整理结果，每个患者目录必须有 `relationships.json`。
原始及整理后的 DICOM 均只读；输出到另一个目录，不包含 NIfTI 转回 DICOM 功能。

## 安装

建议 Python 3.11 或更新版本。以下命令在仓库根目录执行：

```bash
.venv/bin/python -m pip install -r requirements.txt
```

rt-utils 会安装 opencv-python 等依赖。Windows 将解释器替换为 `.venv/Scripts/python.exe`，命令写成一行即可。不需要 PyTorch、GPU 或 OneKey 授权。
如需独立验证，再安装 `SimpleITK`。压缩 DICOM 可能还需要相应 pydicom 解码插件；解码失败会报错，不会生成替代像素。

## 使用

单患者：

```bash
.venv/bin/python convert_dicom_to_nifti.py /path/to/organized_dicom/Case001 \
  --mode patient --output-root /path/to/nifti_data
```

批量预览，不写文件：

```bash
.venv/bin/python convert_dicom_to_nifti.py /path/to/organized_dicom \
  --mode batch --list-only
```

批量转换，并明确允许视野外轮廓裁剪到 CT 网格：

```bash
.venv/bin/python convert_dicom_to_nifti.py /path/to/organized_dicom \
  --mode batch --clip-rois-to-ct --output-root /path/to/nifti_data
```

默认输出到脚本上一级的 `nifti_data/`，建议使用 `--output-root` 明确指定。
使用 `--output-root "其他输出目录"` 可修改。`--mode auto` 是默认值：入口包含 relationships.json 为单患者，否则遍历直接子患者目录。患者内部 DICOM 路径根据索引读取，不受其子目录层级影响；原始未整理文件夹需要先运行 organizer。

- 默认仅将 `DoseSummationType=PLAN` 的剂量转换为 NIfTI，其他剂量仍保留 JSON。
- 加 `--include-beam` 可同时转换 BEAM；不累加剂量，不把 BEAM 当成计划总剂量。
- 默认拒绝超出 CT 视野的 ROI。`--clip-rois-to-ct` 明确授权生成该结构在 CT 网格内的部分；ROI JSON 标记 `clipped_to_ct_grid` 和视野外轮廓点数，报告列出 `clipped_rois`。不能将裁剪后的体积当成原始完整体积。
- 输出已存在且输入、参数、版本和文件校验一致时复用。不同则拒绝覆盖，请换输出目录；包括对已有结果启用 `--include-beam` 或裁剪选项。
- 每例先在临时目录生成再落盘。某个 ROI/剂量失败时其他对象仍可保留，`index.json` 标记 partial 并列出 issues；退出码 1 表示有错误，0 表示没有报告错误。

## 输出结构

```text
nifti_data/
  Case001/
    index.json
    images/
      CT1.nii.gz
      CT1.json
    structures/
      RS1/
        RS1.json
        Heart.nii.gz
        Heart.json
        ...
    plans/
      V15F.json
      V5F.json
    dose/
      V15F_PLAN.nii.gz
      V15F_PLAN.json
      V5F_PLAN.nii.gz
      V5F_PLAN.json
      ...
  conversion_summary_时间戳.json
```

结构名称以实际 ROIName 为准，非法文件名字符会替换，重名附加编号。CT 系列名称沿用整理结果，例如 CT1、FBCT1、CBCT1；RP 优先使用 RTPlanLabel。

## 数值和对应关系

- CT：每层应用 RescaleSlope/RescaleIntercept，float32 HU，按空间位置排序。规则单帧 CT，不做窗宽窗位处理，不重采样。CBCT/FBCT 仍需确认设备标定，应用 DICOM 映射不代表具有可靠的定量 HU。
- RS：每个闭合 ROI 单独 uint8 mask，背景 0、结构 1，允许不同结构重叠；与其引用 CT 同尺寸、原点、方向和间距。定位点/开放轮廓等只保留元数据，不伪造成体积。完整轮廓坐标保留在 RS1.json 的 DICOM JSON 中。
- RP：计划不是体数据，只生成 JSON，保留分次、射野、控制点和引用结构等信息。
- RD：仅接受 DoseUnits=GY，将像素乘 DoseGridScaling 一次，保存 float32 实际 Gy。JSON 写明 dose_units=Gy、dose_grid_scaling_applied=true；NIfTI 不再附加剂量缩放。RELATIVE 不自动猜换算；不自动乘分次数，PLAN 类型本身也不保证是完整疗程，须核对计划。
- CT/RD 保持各自网格，NIfTI affine 为 RAS 毫米坐标，DICOM 原坐标为 LPS。两者不能直接按数组下标叠加。剂量组学如需 CT/结构网格，需要后续显式配准核查及剂量重采样，本脚本不做。
- index.json 汇总 SOP/Series UID、相对文件路径、原文件 SHA256、输出 SHA256、软件版本、选项和问题。RP 引用 RS，RD 引用 RP，ROI 引用 CT；关联使用 UID，不能只凭文件名或目录编号。
- 每类自己的 JSON 保留精选相关 DICOM 标签，嵌套序列用 DICOM JSON（十六进制标签键、vr、Value）存储，不是完整 DICOM 备份。原 DICOM 必须保留。JSON/目录仍可能包含患者信息、UID、日期和自由文本，不是脱敏数据。

## 限制及验证

当前拒绝单层影像、非均匀间距/倾斜剪切 CT、非均匀剂量帧间距、未知剂量单位；不支持增强多帧 CT。为了规避 rt-utils 1.2.7 的网格限制，mask 转换只支持方形 CT 切片及平面内等距像素，轮廓必须唯一引用可用 CT 层。无引用、跨系列引用或平面不一致会报错。轮廓栅格化的边界、孔洞和小结构需与 TPS/SlicerRT 复核，尚未做 TPS 体积/DVH 数值一致性验证。

```bash
PYTHONPATH="$PWD:$PWD/viewer" .venv/bin/python -m unittest \
  tests.test_convert_dicom_to_nifti -v
.venv/bin/python verify_nifti_conversion.py \
  /path/to/organized_dicom /path/to/nifti_data
```

第一项使用合成数据验证排序、HU、剂量缩放、坐标方向及裁剪保护。第二个脚本用独立 SimpleITK 读回真实结果，逐体素核对 CT/剂量、核对物理位置和 mask 网格，校验原文件未改动。科研工具，不用于直接临床决策。

OneKey 后续使用通常选择对应 CT + 一个目标 mask，再按具体 notebook 的目录和标签表要求准备任务输入；无需把所有 CT/ROI/剂量混在同一任务目录。当前转换结果是保留关系的中间数据集，不自动生成任务训练集。

参考：[NIfTI/RAS 坐标](https://nipy.org/nibabel/coordinate_systems.html)、[rt-utils](https://github.com/qurit/rt-utils)、[DICOM 剂量帧位置](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.8.3.2.html)。
