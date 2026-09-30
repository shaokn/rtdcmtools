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
| `server_new.py` | Viewer 的唯一入口，支持 DICOM、NIfTI、快速查看和临时病例库；`--host` 控制监听范围。 |
| `run_viewer_new.sh` | Linux 下启动 Viewer；`RTDCMTOOLS_HOST` 指定绑定地址。 |
| `run_viewer_new_demo.bat` | Windows 启动示例；用户需填写本机 `python.exe` 路径。 |
| `viewer/server.py` | Viewer 核心后端，以库的形式提供 `/api` 接口和静态资源，由 `server_new.py` 加载。 |

`viewer/server.py` 不单独运行，也不自带页面：它只提供接口，页面和 DVH 指标解析都在 `server_new.py` 一侧。`tests/` 中除两个跨数据源一致性用例需要本机病例数据外，其余使用合成 DICOM/NIfTI 数据测试（见「测试」一节）。

## 安装

建议使用 Python 3.11 或更新版本：

```bash
git clone <repository-url>
cd rtdcmtools
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows 将解释器替换为 `.venv\Scripts\python.exe`。Linux 启动脚本依次查找环境变量 `RTDCMTOOLS_PYTHON`、仓库内 `.venv` / `.pyauto`、当前 `aiwork/.pyauto`，最后回退到 `python3`；另可用 `RTDCMTOOLS_HOST` 指定绑定地址（默认 `127.0.0.1`）。

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

Viewer 支持三切面、结构轮廓、剂量色洗、等剂量线、DVH 和 DICOM 关系查看。它默认只监听 `127.0.0.1`，不会主动把数据发送到外网。

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

### 局域网访问（可选）

默认只监听本机回环地址。要让同一网络内的其他电脑打开，显式指定绑定地址：

```bash
python server_new.py --host 0.0.0.0 --port 7777
```

`0.0.0.0` 表示监听所有网卡，它本身**不是浏览地址**。启动时程序会打印真正可用的地址：

```text
New viewer (local): http://127.0.0.1:7777/?source=dicom
New viewer (LAN):   http://192.168.1.50:7777/?source=dicom
```

其他电脑用其中的 LAN 地址打开。也可以只绑某一个网卡（`--host 192.168.1.50`），但网卡地址变化后要同步改命令，一般直接用 `0.0.0.0` 更省事。绑 `0.0.0.0` 不影响本机继续用 `127.0.0.1` 访问。

用启动脚本时通过环境变量传入，脚本的位置参数不变：

```bash
RTDCMTOOLS_HOST=0.0.0.0 ./run_viewer_new.sh 7777 /path/to/organized_dicom /path/to/nifti_fractions
```

Windows 的 `run_viewer_new_demo.bat` 不解析命令行参数，端口和绑定地址都在文件里改，共两处：把顶部的 `PORT` 改成目标端口，再给启动命令加上 `--host 0.0.0.0`。

```bat
set "PORT=7777"
"%PYTHON_EXE%" -u "%~dp0server_new.py" --host 0.0.0.0 --port %PORT% --default-source dicom
```

⚠️ 第二处要改的是**以 `"%PYTHON_EXE%"` 开头的那条启动命令**——它后面还有 `echo` / `pause` / `endlocal` 等几行，别把参数加到文件最末尾。

两点必须注意：

- **端口还要操作系统防火墙放行，本项目不修改防火墙配置。** 例如 ufw：`sudo ufw allow from <你的网段> to any port 7777 proto tcp`。放行前先确认两台机器在同一网段，别用 `ufw allow 7777` 这类不限来源的写法。
- **Viewer 没有任何登录或权限控制。** 端口一旦对局域网开放，能连上的人就能读取所有已加载病例。只在可信网络内短时开启，用完即停。

### Windows 启动

先打开 `run_viewer_new_demo.bat`，将文件顶部的 `PYTHON_EXE` 改成实际的 `python.exe` 路径，例如：

```bat
set "PYTHON_EXE=C:\path\to\rtdcmtools\.venv\Scripts\python.exe"
```

保存后双击 BAT 文件，或在命令提示符中执行：

```bat
run_viewer_new_demo.bat
```

BAT 使用自身所在目录定位 `server_new.py`，因此不要求仓库位于固定盘符或固定文件夹。

端口由文件顶部的 `PORT` 决定，绑定地址由启动命令上的 `--host` 决定，两者都直接在文件里改；要开放局域网访问，见上一节「局域网访问（可选）」。

### 自定义 DVH 指标

DVH 页签的「自定义指标」输入框接受绝对剂量型指标，用逗号、分号或空格分隔，最多 12 项：

| 写法 | 含义 |
| --- | --- |
| `D95%` | 95% 体积接受的剂量。`%` 修饰的是**体积**（规范写法，`D95` 这种省略写法会被拒绝并提示） |
| `D2cc` | 最热 2 cc 体积内的最低剂量 |
| `V20Gy` | 接受 ≥20 Gy 的体积百分比（写成 `V20` 等价） |
| `Dmean` `Dmax` `Dmin` | 平均 / 最大 / 最小剂量 |
| `volume` | 结构体积 cm³ |

留空即使用默认的 `Dmean,D95%,D2%,V20Gy`，取值与升级前的固定列逐位一致——`D_x%` 与内置列共用同一个分位估计（线性插值）。导出 CSV 的列跟随输入框。

两点口径说明：

- `D_xcc` 是「最热 x cc 体积内的**最低**剂量」。当 x cc 小于单个体素时退化为最热体素剂量；当 x cc 超过结构体积时结果等于 `Dmin`。两种情况都会在该行给出提示。
- `V_x%` 是另一种量：「接受 ≥ x% 处方剂量的体积」，需要处方剂量。本工具不携带处方剂量、也不打算支持这类指标，因此这种写法返回 400 并说明原因，**不会**回退成按最大剂量解释；请改用绝对剂量阈值，如 `V20Gy`。

该功能由 `server_new.py` 覆盖核心的 `/api/dvh` 视图函数实现，核心后端 `viewer/server.py` 自身不需要为新增指标改动。

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

测试覆盖 CT 排序与 HU、NIfTI 坐标、剂量缩放、轮廓栅格化保护、Viewer API、快速查看、等剂量显示和自定义 DVH 指标。测试通过不等同于临床验证；仍需针对设备和 TPS 导出格式验证体积、DVH、剂量和空间位置。

⚠️ 其中 `test_server.py::test_all_five_cases_and_roi_projection` 与 `test_nifti_viewer.py::test_all_cases` 要求仓库同级目录下已存在本机的 `organized_dicom`（5 例）与 `nifti_data`，并且会写回 `viewer/qa/data_checks.json`；在没有这些数据的机器上这两个用例会失败，其余用例（含 `test_dvh_metrics.py`）不依赖真实病例。

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
