# RT DICOM Tools

用于放疗科研数据的本地命令行工具和只读影像查看器。工具覆盖以下流程：

```text
DICOM 导出
  -> 伪匿名化（可选）
  -> 按 CT / RTSTRUCT / RTPLAN / RTDOSE / REG 整理
  -> 转换为 CT、ROI mask 和剂量 NIfTI
  -> 在本地浏览 CT、结构、剂量和 DVH
```

本仓库不提供患者数据，也不应提交 DICOM、NIfTI、运行日志或真实病例验证结果。

> **仅供科研和数据核查。** 本项目不是医疗器械，不能替代 TPS、SlicerRT 或临床质控。匿名化脚本只做有限字段的伪匿名化，不构成符合 HIPAA、GDPR 或其他法规的完整 DICOM 去标识化。

## 功能

| 工具 | 用途 |
| --- | --- |
| `anonymize_dcm_dirs.py` | 保持目录结构，替换 `PatientID`、`PatientName` 并清空 `InstitutionName`。 |
| `organize_dicom.py` | 整理普通 DICOM-RT 导出，并用 `relationships.json` 保存 UID 关系。 |
| `organize_dicom_fractions.py` | 将计划 CT 和逐次 FBCT 整理成 `plan_ct1`、`fraction_fbctX`。 |
| `convert_dicom_to_nifti.py` | 将普通整理结果转换为 CT、结构 mask、剂量 NIfTI 和 JSON。 |
| `convert_dicom_fractions_to_nifti.py` | 保留逐次层级，将分次 DICOM-RT 转换为 NIfTI。 |
| `verify_nifti_conversion.py` | 独立读回转换结果，检查几何、数值、关系和校验和。 |
| `server_new.py` | 本地只读 Viewer，支持 DICOM、NIfTI、快速查看和临时病例库。 |
| `run_viewer_new.sh` | Linux 下启动新版 Viewer。 |

`viewer/` 保存 Viewer 后端与静态页面，`tests/` 使用合成 DICOM/NIfTI 数据测试，不依赖真实病例。

## 安装

建议使用 Python 3.11 或更新版本：

```bash
git clone <repository-url>
cd rtdcmtools
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows 将解释器替换为 `.venv\Scripts\python.exe`。Linux 启动脚本依次查找环境变量 `RTDCMTOOLS_PYTHON`、仓库内 `.venv` / `.pyauto`、当前 `aiwork/.pyauto`，最后回退到 `python3`。

## 快速开始

以下示例均在仓库根目录执行，输入和输出必须是不同目录。

### 1. 伪匿名化

```bash
.venv/bin/python anonymize_dcm_dirs.py \
  /path/to/raw_dicom /path/to/pseudonymized Case
```

脚本不会修改源文件。请在进入公开研究流程前使用专业 DICOM 去标识化工具复查私有标签、日期、UID、自由文本和像素内烧录信息。

### 2A. 整理普通 DICOM-RT

```bash
.venv/bin/python organize_dicom.py /path/to/pseudonymized \
  --mode auto --output-root /path/to/organized_dicom
```

先检查、不写文件：

```bash
.venv/bin/python organize_dicom.py /path/to/pseudonymized --mode auto --list-only
```

### 2B. 整理计划 CT 与逐次 FBCT

```bash
.venv/bin/python organize_dicom_fractions.py /path/to/one_patient \
  --output-root /path/to/organized_fractions
```

输出示意：

```text
<PatientID>/
  plan_ct1/CT/ RS/ RP/ RD/
  fraction_fbct1/CT/ RS/ RP/ RD/ REG/
  fraction_fbct2/CT/ RS/ RP/ RD/ REG/
  relationships.json
```

CT/FBCT 类型由文件名和直接父目录提示识别；RS、RP、RD、REG 使用 DICOM UID 引用关系归属。无法唯一归属时停止处理，不自动猜测。

### 3A. 普通数据转换为 NIfTI

```bash
.venv/bin/python convert_dicom_to_nifti.py /path/to/organized_dicom \
  --mode auto --output-root /path/to/nifti_data
```

### 3B. 分次数据转换为 NIfTI

```bash
.venv/bin/python convert_dicom_fractions_to_nifti.py \
  /path/to/organized_fractions/<PatientID> \
  --output-root /path/to/nifti_fractions
```

默认不生成超出 CT 视野的 ROI mask。确实需要保留视野内部分时显式添加 `--clip-rois-to-ct`；生成结果会标记 `clipped_to_ct_grid`，不能将其当成完整结构体积。

### 4. 验证转换

```bash
.venv/bin/python verify_nifti_conversion.py \
  /path/to/organized_dicom /path/to/nifti_data
```

## 本地 Viewer

默认以空病例库启动，DICOM 和 NIfTI 均不预加载任何目录：

```bash
./run_viewer_new.sh 8768
```

打开：

- DICOM：<http://127.0.0.1:8768/?source=dicom>
- NIfTI：<http://127.0.0.1:8768/?source=nifti>

Viewer 支持三切面、结构轮廓、剂量色洗、等剂量线、DVH 和 DICOM 关系查看。它只监听 `127.0.0.1`，不会主动把数据发送到外网。

左侧 `+` 可选择文件夹：

- DICOM 会递归扫描并按 `PatientID` 拆分病例，兼容平铺和嵌套目录。
- NIfTI 会递归识别转换结果中的 `index.json`。
- 浏览器会把所选文件复制到本机临时会话目录；停止服务后自动清理。
- 病例右上角 `x` 只从当前列表移除，不删除源目录。

大型数据更适合在启动命令中直接传入根目录，以避免浏览器临时复制。按 `Ctrl+C` 停止服务。

确实需要预加载时，可显式传入 DICOM 和 NIfTI 根目录：

```bash
./run_viewer_new.sh 8768 /path/to/organized_dicom /path/to/nifti_fractions
```

路径不会作为默认值写在 Python 程序中；未传入的类型保持空列表。

## NIfTI 输出约定

- **CT**：逐层应用 `RescaleSlope/RescaleIntercept`，输出 `float32` HU。
- **RS**：每个可栅格化 ROI 输出一个二值 mask，与引用 CT 使用相同网格。
- **RP**：RTPLAN 不是体数据，保留原 DICOM，并输出 JSON 元数据。
- **RD**：应用一次 `DoseGridScaling`，输出 `float32`，单位为 Gy；不自动乘分次数。
- **坐标**：NIfTI 使用 RAS 毫米坐标；原 DICOM 使用 LPS。CT 与 RD 保留各自网格，不能直接按数组下标叠加。
- **关系**：`index.json` / `relationships.json` 使用 SOP、Series 和 Frame of Reference UID 建立关联，不依赖文件名猜测。

分次转换结果示意：

```text
<PatientID>/
  plan_ct1/
    CT/ct.nii.gz
    RS/RS1/metadata.json + masks/*.nii.gz
    RP/RP1.dcm + RP1.json
    RD/PLAN.nii.gz + PLAN.json
    index.json
  fraction_fbct1/
    CT/ct.nii.gz
    RS/RS1/metadata.json + masks/*.nii.gz
    RP/RP1.dcm + RP1.json
    RD/PLAN.nii.gz + PLAN.json
    REG/REG1.json
    index.json
```

## 测试

```bash
PYTHONPATH="$PWD:$PWD/viewer" .venv/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v
```

测试覆盖 CT 排序与 HU、NIfTI 坐标、剂量缩放、轮廓栅格化保护、Viewer API、快速查看和等剂量显示。测试通过不等同于临床验证；仍需针对设备和 TPS 导出格式验证体积、DVH、剂量和空间位置。

## 已知限制

- 有限字段伪匿名化不处理全部 DICOM PHI，也不检测像素内烧录文字。
- 不支持所有增强多帧、非均匀层间距、倾斜剪切或厂商私有格式。
- FBCT/CBCT 应用 DICOM 像素映射不代表具有可靠定量 HU。
- ROI 栅格化、剂量重采样和 DVH 应与 TPS 或 SlicerRT 交叉核对。
- Viewer 不自动跨 CT 应用 REG，也不执行形变剂量累积。

更详细的转换说明见 [docs/convert_dicom_to_nifti_usage.md](docs/convert_dicom_to_nifti_usage.md)。

## 公开发布检查

仓库已通过 `.gitignore` 排除 DICOM、NIfTI、模型、日志、缓存和本地 QA 结果。首次推送前仍应运行：

```bash
git status --ignored
git grep -nE '(/Users/|/home/[^ <]+|[A-Za-z]+[0-9]{8,})' -- ':!README.md'
```

## 许可证

本项目采用 [MIT License](LICENSE)。
