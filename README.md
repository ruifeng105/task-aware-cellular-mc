# 论文初稿与仿真合并包

**History-Aware Learning for Task-Aware Cellular Molecular Communication**  
整理日期：2026-10-04。

本包把七页英文论文初稿、LaTeX 源码、真实数据学习试验、响应模型拟合、作者仿真参照和结果核查放在同一项目中。可先打开 `paper/AI_MC_Cellular_Receivers_Draft.pdf` 阅读论文，再按下方入口复现。中文研究说明见 `Paper_Notes_ZH.md`，论文与程序的逐项对应见 `Paper_Simulation_Map_ZH.md`。

## 快速运行

在解压后的 `task-aware-cellular-mc/` 中运行：

```bash
python -m pip install -r requirements.txt
python scripts/reproduce.py --mode verify
```

`verify` 检查 47 个作者来源文件、输入时序、历史特征的因果性、混合条件归一化及保存预测的评分，并将数值与论文对应的固定参照比较。它不重新训练。

完整重跑已经规定的模型拟合、预测、核查和论文图表：

```bash
python scripts/reproduce.py --mode full
```

命令会更新 `simulation/results/fgf2_pilot/` 和论文图表，输出简短状态；详细日志及运行报告在 `reproduction/`。脚本也可从其他目录以绝对路径调用。打包时的固定参照保存在 `reproduction/reference_metrics.json`；修改模型后不要覆盖它来让核查通过。

## 内容与完成状态

| 路径 | 内容 |
|---|---|
| `paper/AI_MC_Cellular_Receivers_Draft.pdf` | 已编译的七页英文初稿 |
| `paper/main.tex`、`paper/references.bib`、`paper/IEEEtran.cls` | LaTeX 源码、11 篇主要参考文献与排版类文件 |
| `paper/figures/`、`paper/tables/` | 两组结果图和四张表；系统图、算法框位于 `main.tex` |
| `simulation/code/` | 响应模型、预测基线、数据检查与通用分析程序 |
| `simulation/public_data/` | 47 个作者原始导出文件及来源清单 |
| `simulation/results/fgf2_pilot/` | 预测 CSV、响应曲线、匹配分析和数值核查结果 |
| `simulation/novelty_audit.json`、`simulation/AI_MC_Application_First_Review_ZH.md` | 18 项相关工作定向核查、应用标定分析和仍待补齐的全文证据 |
| `simulation/configs/` | 已执行的数据划分，以及尚未执行的生物/工业扩展方案 |
| `scripts/reproduce.py` | 统一核查、复现和可选论文编译入口 |
| `release_manifest.json` | 合并包文件的 SHA-256 与大小清单 |

**已执行算法**是按条件加权的因果历史岭回归：利用截至预测时刻的输入和响应历史，以及共同已知的未来命令，预测单细胞响应。另有简单开环命令到响应模型的拟合与曲线计算。训练、验证、测试使用完整条件块，不随机拆分细胞或滑动窗口。

**B3 参照**重用原作者导出的预测样本及均值。没有在本包中重新执行 B3 参数推断或原始生化仿真器；原研究曾使用这些条件选择模型结构，因此该参照不是新盲测，也不是同反馈信息条件的预测比较。

**拟议任务控制**包含准备度、幅度/持续时间/等待时间优化与序列可靠性约束，目前只有论文算法流程和配置草案。尚无实现或闭环控制结果。新颖性核查保留了全文/补充材料缺口，不能据此宣称已证明开创性或优先权。

## 论文对应的固定结果

数据包含 606 条单细胞轨迹、53,998 个观测点，采样间隔 2 分钟。训练为持续刺激和 3 分钟开/20 分钟关条件，验证为单次 5 分钟刺激，测试为混合刺激。

| 指标 | 固定结果 |
|---|---:|
| 10 分钟预测：因果历史 RMSE | 0.026817 |
| 10 分钟预测：当前输入/响应/时钟 RMSE | 0.029229 |
| 上述 RMSE 的相对改善 | 8.25% |
| 2 分钟预测：因果历史 RMSE | 0.015391 |

这些是按条件等权的点估计。当前数据没有已核实的独立实验日标识，不能把细胞或窗口数当作独立重复数。测试结果已经查看；后续模型开发应使用新的独立实验数据。命令浓度不等于已测局部浓度或释放分子数；预测改善不能证明新生物记忆机制、任务成功率或剂量节省。

## 编译论文

已附 PDF，运行 Python 核查无需 LaTeX。安装含常用宏包的 TeX Live 或 MiKTeX 后，可在项目根目录运行：

```bash
python scripts/reproduce.py --mode verify --compile-paper
```

也可直接进入 `paper/`：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=AI_MC_Cellular_Receivers_Draft main.tex
```

没有 `latexmk` 时，依次运行 `pdflatex`、`bibtex AI_MC_Cellular_Receivers_Draft`、再运行两次 `pdflatex`；各次 `pdflatex` 使用上述相同参数。Overleaf 上传 `paper/` 下的文件并选择 `main.tex`。本稿使用 IEEEtran 1.8b conference 布局，保留页码及初稿提示，尚未按特定投稿会议要求调整。

仅重建论文图表可运行 `python scripts/build_paper_assets.py`；该步骤读取保存结果，不重新训练。

## 环境、来源与署名

固定试验的历史环境记录在 `simulation/results/fgf2_pilot/runtime_versions.json`；本次合并复现的实际环境记录在 `reproduction/run_report.json`。`requirements.txt` 列出 Python 依赖；LaTeX 是可选的独立依赖。程序运行不需要联网下载数据。

如需使用本次复现的精确 Python 包版本，可改用 `python -m pip install -r requirements-reproduced.txt`。本次环境为 Python 3.12.14；版本文件记录 NumPy、pandas、openpyxl 和 matplotlib。

实验时间序列与 B3 仿真导出来自 Blum 等，*Molecular Systems Biology*，2019，DOI `10.15252/msb.20198947`，及论文关联的 `Mijan/LFNS_MSB` 仓库：分支 `MSB_version`，固定提交 `5c917abda0618d75c00c9cab45f24ed893dd71f1`。相关 Mendeley Data 记录为 Maciej Dobrzyński 与 Yannick Blum，v2，2020-05-13，DOI `10.17632/ccnxn84w8z.2`，CC BY 4.0。

本包使用仓库中的数值子集，不包含完整 Mendeley 压缩包或显微图像。保留作者文件名与来源身份；不为作者仓库文件或整个合并包附加统一的新许可。未修改的 `paper/IEEEtran.cls` 为 Michael Shell 的 1.8b 版本，保留版权和 LaTeX Project Public License 声明，来自 `bardsoftware/template-ieee-transactions` 公共镜像。

后续推进顺序：补齐新颖性全文证据与可反驳的 gap → 新独立实验日和局部输运标定 → 强基线与历史增量验证 → 实现并评估任务控制 → 再扩展数据规模和应用场景。
