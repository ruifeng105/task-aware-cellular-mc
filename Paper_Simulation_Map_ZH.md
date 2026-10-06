# 论文与仿真、数据和结果的对应

本表对应随包附带的初稿（2026-10-06 六页修订版，排版后 5 页，三图两表）。所有路径相对于项目根目录。

| 论文内容 | 数据或结果 | 对应程序与状态 |
|---|---|---|
| Sec. II 式 (1)–(2)、Fig. 1：系统与探测任务、准备度定义 | `paper/main.tex` 内的 TikZ 图 | 下一命令的幅度和持续时间固定，只决定发送时刻；任务门槛 $\kappa=0.5$、窗口 20 min、失败惩罚 $D=140$ min。实验数据没有任务标签，只在 B3 上仿真 |
| Sec. III-A 式 (3)–(4)：嵌套预测器 P/C/O/M/H | `simulation/results/nested/nested_results.json` 的 `main` | `nested_forecast.py`（方案 `simulation/configs/nested_forecast_protocol.json`，冻结记录 `results/nested/nested_forecast_protocol_freeze.json`）。C 等于预登记的 IR + slope，H 的特征集合等于旧 M3 |
| Sec. III-B 式 (5)：校准的等待规则 | `simulation/results/b3/waiting_results.json` | `b3_waiting.py`（准备度方案第 3 版 `b3_waiting_protocol.json`，冻结记录 `results/b3/b3_waiting_protocol_freeze.json`）；后验样本按种子 20261006 分成校准组和评价组各 500 个 |
| Table I：协议、角色、细胞数与 B3 用途 | `fgf2_expanded/expanded_results.json` 的 `inventory`；`nested_results.json` 的 `b3_roles` | B3 用途由 `nested_forecast.b3_data_roles` 从作者推断配置 `config_fgf_sus_3_20.xml` 读出（LFNS 拟合实验与 ComputeLikelihood 结构比较实验） |
| Table II：核心模型 RMSE 与逐条件方向 | `nested_results.json` 的 `main` 下 10 分钟和 2 分钟的 `comparisons` | 由 `scripts/build_paper_assets.py` 生成 `paper/tables/core_rmse.tex`；`verify_nested_forecast.py` 从保存的模型重算 90 个分数，并核对 H 与旧 M3 一致 |
| Fig. 2(a)：Test A、25 ng/ml 的示例细胞 | `nested_results.json` 的 `example` | 按方案规则取 10 分钟 H 误差居中的细胞（64 个中的第 16 号） |
| Fig. 2(b)：分阶段误差 | `nested_results.json` 的 `phases` | `nested_forecast.phase`：刺激期、下一命令窗口、早期洗脱（停药后 ≤30 min）、后期；核查脚本在切换时刻做 11 个边界测试 |
| Sec. V-C：修正的留一方案检验 | `nested_results.json` 的 `lopo` | `nested_forecast.lopo_fold`：σ 在训练组上选，惩罚系数用训练组内的留一组交叉验证选；核查脚本把留出组目标值整体 +1，确认所选参数和预测都不变 |
| Fig. 3、Sec. V-D：校准等待、平滑基线与噪声消融 | `waiting_results.json`；Fig. 3(a) 来自 `readiness_results.json` 的 `matched_reporter_readiness` | `verify_b3_waiting.py`：与 v2 估计器等价、只用过去观测、翻转评价组标签后校准选择不变、回退规则、重算评价结果，并给出权重坍缩诊断（有效样本数） |
| Sec. IV-B：B3 执行与验收 | `simulation/results/b3/verification_results.json` | `b3_model.py`、`verify_b3.py`；来源清单 `fgf2_b3_source_manifest.json` |
| 来源、时序与因果性核查 | `fgf2_source_manifest.json`（47 个）、`fgf2_expanded_source_manifest.json`（56 个）、`fgf2_b3_source_manifest.json`（3 个） | `verify_fgf2_pipeline.py`、`verify_expanded_fgf2.py`、`verify_b3.py` |

## 移出正文、保留在复现包中的内容

| 内容 | 文件 | 说明 |
|---|---|---|
| 试点结果（原 Table II） | `paper/tables/forecast_rmse.tex`；`fgf2_pilot/calibration_results.json` | 正文只用一句话交代；测试集已查看 |
| 预登记扩展的原始分析（原 Table III） | `paper/tables/expanded_rmse.tex`；`expanded_results.json` | 正文保留 H1 成立（8.8%）以及相对不变预测只好 0.7% 的披露 |
| 原 LOPO（惩罚系数沿用主分析，存在调参泄漏） | `paper/tables/lopo_rmse.tex`；`expanded_results.json` 的 `leave_one_protocol_out` | 正文报告修正后的结果，并说明旧结果的问题 |
| 开环模型与 B3 参照（原 Table V） | `paper/tables/open_loop_rmse.tex` | 无反馈的群体均值预测，与核心问题距离较远 |
| 探索性历史匹配（原 VI-G） | `paper/tables/matched_history.tex`；`exploratory_matched_history_pairs.csv` | 不能识别因果历史效应 |
| 准备度 v1/v2 结果与原 Fig. 3 | `readiness_results_v1.json`、`readiness_results.json`；`paper/figures/readiness_policies.*` | v2 的阈值和固定等待在同一批样本上选取并评价，正文改用 v3 结果 |
| 单样本等待示例（原 Fig. 2） | `readiness_example.json`；`paper/figures/readiness_example.*` | 由 `b3_readiness_example.py` 生成，作为补充材料 |
| MPC 式与完整前瞻验证路线 | `simulation/configs/prospective_receiver_reuse.json`、`validation_kit/`、`AI_MC_Complete_Validation_Plan_ZH.md` | 正文只保留等待型复用 |

## 使用统一入口

`python scripts/reproduce.py --mode verify` 核查保存的预测、扩展检验、B3 执行、B3 预测与准备度（v2 和单样本示例）、嵌套对照与修正 LOPO、校准等待研究，以及固定论文数值（需要 SciPy，约 6 分钟）。`--mode full` 从随包数据重跑全部分析（B3 准备度仿真约 5 分钟，嵌套对照约 2 分钟），再核查并重建图表与规划工具输出。两种模式都不增加训练数据、不重新推断 B3 参数，也不在真实接收器上执行任务控制。

固定参照在 `reproduction/reference_metrics.json`：原 69 项冻结值保持不变，新增条目及原因写在 `extensions` 字段中，共 862 项。`pipeline_checks.py` 检查正文中所有四位以上小数都在固定参照中，并检查正文引用的每个表、图都存在。`reproduction/run_report.json` 记录本次执行状态和环境，日志位于同目录的 `logs/`。

`release_manifest.json` 和 `simulation/checksums.json` 描述打包时的文件。所有生成文件都写成 LF，`.gitattributes` 保证在任何平台上检出的文本文件与清单逐字节一致。在不同库版本下重跑，结果的最后几位浮点数可能变化，论文数值以固定参照比较为准。
