# 论文初稿与仿真合并包

**Toward Task-Aware Reuse of Cellular Molecular Receivers: History-Aware Prediction on Experimental FGF2–ERK Data**
整理日期：2026-10-04（ICC 投稿修订版）。

本包把五页英文论文初稿、LaTeX 源码、真实数据学习试验、预先登记的扩展检验、作者仿真参照和结果核查放在同一项目中。可先打开 `paper/AI_MC_Cellular_Receivers_Draft.pdf` 阅读论文，再按下方入口复现。中文研究说明见 `Paper_Notes_ZH.md`，论文与程序的逐项对应见 `Paper_Simulation_Map_ZH.md`。

## 快速运行

需要 Python ≥ 3.10。在项目根目录运行：

```bash
python -m pip install -r requirements.txt
python scripts/reproduce.py --mode verify
```

`verify` 不重新训练，依次核查：试点的 47 个作者来源文件、输入时序、特征因果性、混合条件归一化和保存评分；扩展检验的 56 个新增来源文件、冻结方案哈希、10 分钟单脉冲时序（对照 XML）、60 分钟单脉冲时序推断、特征因果性和从保存系数重算的 70 个评分；合成数据单元检查；最后与 416 项论文数值固定参照逐项比较，确认正文中所有四位以上小数都在固定参照中，并检查生成文件均为 LF 换行。

完整重跑模型拟合、扩展检验、核查和论文图表：

```bash
python scripts/reproduce.py --mode full
```

命令会更新 `simulation/results/` 和论文图表，详细日志及运行报告在 `reproduction/`。固定参照保存在 `reproduction/reference_metrics.json`：原 69 项保持冻结值，新增条目及原因记录在其 `extensions` 字段；修改模型后不要覆盖它来让核查通过。

仓库的 `.gitattributes` 强制文本文件使用 LF。来源文件按 Git blob 和 SHA-256 哈希核验，Windows 默认的 CRLF 转换会让全新克隆的核验全部失败，因此不要删除该文件。

## 内容与完成状态

| 路径 | 内容 |
|---|---|
| `paper/AI_MC_Cellular_Receivers_Draft.pdf` | 已编译的五页英文初稿 |
| `paper/main.tex`、`paper/references.bib`、`paper/IEEEtran.cls` | LaTeX 源码、11 篇参考文献（IEEEtran 文献样式）与排版类文件 |
| `paper/figures/`、`paper/tables/` | 结果图和五张表，均由脚本从保存结果生成 |
| `simulation/code/` | 试点（`calibrate_fgf2.py`）、扩展检验（`expanded_fgf2.py`）、核查、合成单元检查与通用分析程序 |
| `simulation/configs/fgf2_expanded_protocol.json` | 下载新数据前冻结的扩展检验方案 |
| `simulation/results/fgf2_pilot/`、`simulation/results/fgf2_expanded/` | 试点与扩展检验的结果、冻结记录和核查报告 |
| `simulation/public_data/` | 作者原始导出文件；试点 47 个与扩展 56 个分别列在两份来源清单中 |
| `simulation/novelty_audit.json`、`simulation/AI_MC_Application_First_Review_ZH.md` | 18 项相关工作定向核查与应用标定分析 |
| `scripts/reproduce.py` | 统一核查、复现和可选论文编译入口 |
| `release_manifest.json` | 合并包文件的 SHA-256 与大小清单 |

**已执行算法**是按条件加权的因果历史岭回归：利用截至预测时刻的输入和响应历史，以及各模型共同已知的未来命令，预测单细胞响应；另有"历史 + 时钟"嵌套对照、开环命令到响应模型和训练均值常数基线。划分使用完整条件块，不随机拆分细胞或滑动窗口。

**B3 参照**重用原作者导出的预测样本均值，没有重新执行 B3 参数推断。原研究曾用这些条件选择模型结构，因此它不是新盲测。

**拟议任务控制**（准备度、幅度/持续时间/等待时间优化、序列可靠性约束）目前只有算法流程和配置草案，没有实现或闭环结果。

## 论文对应的主要结果

**试点**（606 条轨迹、53,998 个观测点；测试集此前已查看）：10 分钟预测中，因果历史 RMSE 0.026817，当前输入/响应/时钟为 0.029229，相对改善 8.25%。但 42.5% 的测试预测起点晚于最后一个训练起点；在这一段，"不变预测"最好。验证集上历史模型只比"不变预测"好 1.6%。

**预先登记的扩展检验**（22 个条件、1,565 条轨迹；方案在下载 14 个新条件之前冻结，分析只运行一次）：

| 10 分钟 RMSE | 不变预测 | 当前响应 + 时钟 | 因果历史 | 历史 + 时钟 |
|---|---:|---:|---:|---:|
| A：10 分钟单脉冲（新方案） | 0.02287 | 0.02490 | 0.02270 | 0.02286 |
| B：60 分钟单脉冲（新方案，时序推断） | 0.02651 | 0.02709 | 0.02568 | 0.02624 |
| C：混合刺激 0.25/25 ng/ml（新浓度，与已看测试同一批次） | 0.02876 | 0.02792 | 0.02778 | 0.03021 |

主要假设 H1、H2 成立：在 A 上，历史模型比当前响应 + 时钟降低 8.8%，历史 + 时钟降低 8.2%。但相对"不变预测"只好 0.7%，且只在 4 个浓度中的 1 个更好。留一方案检验中，用 5 类方案训练时，历史特征在 6 个留出方案上 RMSE 都最低，比"不变预测"好 5.8%–13.6%。

这些都是按条件等权的点估计。数据没有独立实验日标识，不能把细胞或窗口数当作独立重复数，也不报告显著性。命令浓度不等于已测局部浓度；预测改善不能证明新生物记忆机制、任务成功率或剂量节省。

## 编译论文

已附 PDF，运行 Python 核查无需 LaTeX。安装 TeX Live 或 MiKTeX 后，可在项目根目录运行：

```bash
python scripts/reproduce.py --mode verify --compile-paper
```

也可在 `paper/` 中运行 `latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=AI_MC_Cellular_Receivers_Draft main.tex`。没有 `latexmk` 时，依次运行 `pdflatex`、`bibtex AI_MC_Cellular_Receivers_Draft`，再运行两次 `pdflatex`。Overleaf 上传 `paper/` 下的文件并选择 `main.tex`。本稿使用 IEEEtran 会议版式，共 5 页；投稿前请按目标会议的征稿要求核对页数与格式。

仅重建论文图表可运行 `python scripts/build_paper_assets.py`，该步骤读取保存结果，不重新训练。

## 环境、来源与署名

固定试点的历史环境记录在 `simulation/results/fgf2_pilot/runtime_versions.json`，本次复现的实际环境记录在 `reproduction/run_report.json`。程序运行不需要联网；`simulation/code/fetch_expanded_sources.py` 只用于重新下载并核验扩展检验的来源文件，是可选步骤。

实验时间序列与 B3 仿真导出来自 Blum 等，*Molecular Systems Biology*，2019，DOI `10.15252/msb.20198947`，以及论文关联的 `Mijan/LFNS_MSB` 仓库（分支 `MSB_version`，固定提交 `5c917abda0618d75c00c9cab45f24ed893dd71f1`）。相关 Mendeley Data 记录为 Maciej Dobrzyński（贡献者 Yannick Blum），v2，2020-05-13，DOI `10.17632/ccnxn84w8z.2`，CC BY 4.0。

本包使用仓库中的数值子集，不包含完整 Mendeley 压缩包或显微图像。作者文件保留原文件名与来源身份。作者仓库根目录没有代码许可证，本包不为这些文件或整个合并包附加新的许可；若公开发布仓库，请先确认这些文件可以再分发。未修改的 `paper/IEEEtran.cls` 为 Michael Shell 的 1.8b 版本，保留其 LaTeX Project Public License 声明。

后续推进顺序：投稿版定稿 → B3 数字孪生（复现作者 `sim_post` 导出作为验收）→ 仿真评估任务控制 → 新独立实验日和局部输运标定 → 活细胞控制实验。
