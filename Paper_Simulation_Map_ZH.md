# 论文与仿真、数据和结果的对应

本表对应随包附带的初稿（2026-10-08 版：在 2026-10-07 补强版上加入模型失配、0.94 条件图和下一命令窗口拆分；目前 9 页含参考文献，四图四表）。所有路径相对于项目根目录。正文使用描述性预测器名称，与代码键的对应为：Persist = P、Current = C、Filter = O、Lag = A、Filter+B3 = M、Lag+B3 = A+M、F+B3+Input = H、F+B3 (phase) = M_phase。

| 论文内容 | 数据或结果 | 对应程序与状态 |
|---|---|---|
| Sec. I 引言：MC 定位（带记忆接收机、保护间隔/检测区间、序贯检测与最优停止） | `paper/references.bib` 新增 kilinc2013、mosayebi2014、cao2020、tung2019、wald1945、poor2009 | 只按摘要层面引用；均经 Crossref 核对并附 DOI（原 tung2018 的题目在 Crossref 查不到：Tung & Mitra 在 ISTC 2018 的论文实为 "Increasing Robustness to Synchronisation Errors in Molecular Communications"，doi 10.1109/ISTC.2018.8625364；改引更贴合序贯检测的 ICC 2019 DFE-SPRT 论文） |
| Sec. II 式 (1)–(3)：系统、任务、逐历史约束的设计问题；κ 与 $T_{\rm f}$ 的依据 | `simulation/results/b3/waiting_frontier_results.json` 的 `kappa_feasibility`、`summary/penalty` | `b3_waiting_frontier.py`（方案 `b3_waiting_frontier_protocol.json`）：κ = 0.5 是 oracle 成功率仍 > 99% 的最大取值；$T_{\rm f}=140$ 为任何成功的最晚完成时间，160–260 为敏感性 |
| Table I：预测器特征表 | 正文内 | 特征定义见 `nested_forecast.nested_features`、`nested_arx.py`；细节见 `paper/supplementary_details.md` S2 |
| Sec. III-A 式 (4)–(5)：滤波与 B3 增量 | `simulation/results/nested/nested_results.json` | `nested_forecast.py`（方案 `nested_forecast_protocol.json`） |
| Sec. III-B 式 (6)：等待规则、逐历史校准、B3 预测与"平滑 + 斜率" | `waiting_frontier_results.json`、`delay_map_results.json` | `b3_waiting_frontier.py`、`b3_delay_map.py`（方案 `b3_delay_map_protocol.json`） |
| Sec. IV 式 (7)：评分、噪声调整 skill、细胞 bootstrap | `simulation/results/nested/nested_extensions_results.json` | `nested_extensions.py`（方案 `nested_extensions_protocol.json`）；`verify_nested_extensions.py` 核对 2/10 分钟与冻结结果一致（差 < 1e-10） |
| Table II：协议、角色与 B3 用途 | `fgf2_expanded/expanded_results.json`；`nested_results.json` 的 `b3_roles` | 同上一版 |
| Table III(a)(b)：核心 RMSE 与修正 LOPO | `nested_results.json`、`nested_arx_results.json` | `scripts/build_paper_assets.py` 生成 `core_rmse.tex`、`lopo_core.tex` |
| Fig. 2(b)（原 Table III(c)，2026-10-08 改为折线图；半衰减穿越的平衡准确率改在正文报告）：噪声调整 skill 随预测时域（2/10/20/30 分钟）、Filter+B3 的 95% 区间、半衰减穿越的平衡准确率 | `nested_extensions_results.json` 的 `horizons/*/roles/test_new_protocol` | 生成 `paper/figures/prediction_phases.*` 的 (b) 面板；`skill_horizon.tex` 仍生成，留在复现包中 |
| Fig. 2、Sec. V-B：分阶段误差与相位门控 | `nested_results.json` 的 `phases`；`nested_extensions_results.json` 的 `phases`、`lopo` | 相位门控两种变体只用训练协议拟合；切换规则被训练数据否决，相位权重使下一命令窗口误差升到 0.03571。按后续脉冲拆分（V-B 节末段）见 `simulation/results/nested/onset_split_results.json`：`nested_onset_split.py`（方案 `nested_onset_split_protocol.json`），`verify_nested_onset_split.py` 核对分组、合并后与原窗口一致（差 1.4e-17）和重算 |
| Table IV、Sec. V-C：逐历史约束下的等待与 oracle 差距弥合比例 | `waiting_frontier_results.json` 的 `repeats/0/*/cells/0.94`、`summary` | 生成 `waiting_per_history.tex`；`verify_b3_waiting_frontier.py` 核对逐历史固定等待等于冻结的 per-history 结果、防泄漏与单次重算 |
| Fig. 3(a)(b)：可靠性—时延前沿 | `waiting_frontier_results.json` 的 `repeats/0/0.005/frontier` | 评价集上扫阈值，只作描述；空心点为 0.94 的校准工作点 |
| Fig. 3(c)(d)、Sec. V-D：延迟与条件图（校准目标 0.94） | `delay_map_target_results.json` 的 `summary/0.94`、`headline_model_helps_cells`（加粗格）、`class_changes_090_to_094`；0.90 版 `delay_map_results.json` 保留 | `b3_delay_map_target.py`（方案 `b3_delay_map_target_protocol.json`）；`verify_b3_delay_map_target.py`：0.90 时与原条件图逐位一致，0.94 时 d = 0、Δ = 2 的格子与前沿研究一致，可行性、防泄漏、两次重算；原图的 `verify_b3_delay_map.py` 仍有效 |
| Sec. V-E：真实数据自然探测 | `simulation/results/b3/natural_probe_results.json` | `natural_probe.py`（方案 `natural_probe_protocol.json`）；`verify_natural_probe.py` 单元测试并整体重算 |
| Sec. V-E 后半：B3 恢复的后验预测检查与权重塌缩诊断 | `simulation/results/b3/recovery_check_results.json`（`mixed`、`three_twenty`、`collapse`、`reading`） | `b3_recovery_check.py`（方案 `b3_recovery_check_protocol.json`，在探索之后冻结）；`verify_b3_recovery_check.py`：与冻结的自然探测一致、单元测试、整体重算、判读规则独立重推 |
| Fig. 4：(a) B3 恢复的后验预测检查，(b) 隐藏恢复延迟下长命令后的成功率 | (a) `recovery_check_results.json` 的 `mixed`、`three_twenty`；(b) `waiting_mismatch_results.json` 的 `summary/0.005/hidden_lag/model_calibrated/*/0.94/success/*/long`、`descriptive/shift_equivalents` | `scripts/build_paper_assets.py` 生成 `paper/figures/model_mismatch.*`，只读冻结结果 |
| Sec. V-F：模型失配（隐藏/可见的恢复延迟、按真实结果重新校准、慢恢复子群、实测探针对应的延迟） | `waiting_mismatch_results.json` 的 `statements`（M1–M4）、`descriptive`、`summary`；逐次结果 `waiting_mismatch_splits.json` | `b3_waiting_mismatch.py`（方案 `b3_waiting_mismatch_protocol.json`，主分析为噪声 0.005、目标 0.94）；`verify_b3_waiting_mismatch.py`：基线与前沿研究一致、隐藏延迟不变量、慢尾与重新校准、防泄漏、单次重算、描述性结果，并从逐次结果独立重算 M1–M4 |
| Sec. V-C 末尾提到的早期敏感性分析 | `waiting_sensitivity_results.json`、`waiting_margin_results.json`、`ar1_likelihood_results.json` 等 | 原 Table IV 移到复现包（`paper/tables/waiting_sensitivity.tex`） |
| Sec. V-H：证据边界 | — | 原先分散在各节的限定语集中于此 |

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
| 原 Table III（合并约束下的分历史结果）与原 Table IV（κ、噪声、失配敏感性） | `paper/tables/waiting_history.tex`、`paper/tables/waiting_sensitivity.tex` | 2026-10-07 版改用逐历史约束（新 Table IV），早期敏感性在正文用一句话概括 |
| 原 Fig. 3（四面板：准备度、噪声、成功—完成、20 次划分） | `paper/figures/readiness_waiting.*` | 由新 Fig. 3（前沿 + 延迟条件图）取代 |
| 归一化、滞后填充、B3 执行、校准细节、因果判据、AR(1) 似然 | `paper/supplementary_details.md` | 补强清单第 3 项：从正文移到仓库 |
| MPC 式与完整前瞻验证路线 | `simulation/configs/prospective_receiver_reuse.json`、`validation_kit/`、`AI_MC_Complete_Validation_Plan_ZH.md` | 正文只保留等待型复用 |

## 使用统一入口

`python scripts/reproduce.py --mode verify` 核查保存的预测、扩展检验、B3 执行、B3 预测与准备度（v2 和单样本示例）、嵌套对照与修正 LOPO、校准等待研究，以及固定论文数值（需要 SciPy，约 6 分钟）。`--mode full` 从随包数据重跑全部分析（B3 准备度仿真约 5 分钟，嵌套对照约 2 分钟），再核查并重建图表与规划工具输出。两种模式都不增加训练数据、不重新推断 B3 参数，也不在真实接收器上执行任务控制。

固定参照在 `reproduction/reference_metrics.json`：原 69 项冻结值保持不变，新增条目及原因写在 `extensions` 字段中，共 41895 项。`pipeline_checks.py` 检查正文中所有四位以上小数都在固定参照中，并检查正文引用的每个表、图都存在。`reproduction/run_report.json` 记录本次执行状态和环境，日志位于同目录的 `logs/`。

`release_manifest.json` 和 `simulation/checksums.json` 描述打包时的文件。所有生成文件都写成 LF，`.gitattributes` 保证在任何平台上检出的文本文件与清单逐字节一致。在不同库版本下重跑，结果的最后几位浮点数可能变化，论文数值以固定参照比较为准。
