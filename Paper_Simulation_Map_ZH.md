# 论文与仿真、数据和结果的对应

本表对应随包附带的投稿稿（2026-10-08 按修改更新方案修订：正文 7 页含参考文献，三图三表，`paper/main.tex`；补充材料 9 页，十一表两图，`paper/supplement.tex`，编译为 `paper/AI_MC_Cellular_Receivers_Supplement.pdf`）。所有路径相对于项目根目录。正文使用描述性预测器名称，与代码键的对应为：Persist = P、Current = C、Filter = O、Lag = A、Filter+B3 = M、Lag+B3 = A+M、F+B3+Input = H、F+B3 (phase) = M_phase；等待规则中 B3 belief 的代码键为 `history`。

| 论文内容（正文 / 补充材料） | 数据或结果 | 对应程序与状态 |
|---|---|---|
| 正文 Sec. I：主线与三条贡献；第三条（速度—鲁棒性权衡）的余量数字 | 下列各行；余量见 `waiting_mismatch_splits.json` 的 `repeats/*/0.005/slack/0.94/*/long/quantiles`（20 次划分取中位数） | `scripts/reproduce.py` 的 `revision8_metrics` 键 `b3/mismatch_slack/...` |
| 正文 Sec. I 相关工作 | `paper/references.bib` | 27 篇参考文献；其中 21 篇经 Crossref 核对，cheng2024、cai2025、borges2024、dong2026 来自原 bib 文件，clopper1934、angelopoulos2021 为 2026-10-08 新增，以上 6 篇尚未逐条核对，投稿前建议再查 |
| 正文 Sec. II 式 (1)–(3)、图 1：任务、代理任务说明、抽样单位 | `b3_waiting_strict_protocol.json` 的 `sampling_unit`；`waiting_scope_results.json`（κ、W、失败惩罚） | `b3_waiting_strict.py`、`b3_waiting_scope.py` |
| 正文 Sec. III-A 式 (4)–(5)：嵌套预测器与 B3 增量；补充表 S1 | `simulation/results/nested/nested_results.json` | `nested_forecast.py`、`nested_arx.py` |
| 正文 Sec. III-B 式 (6)–(7)：阈值规则族、参考/校准/测试角色、固定序列认证、回退 | `waiting_strict_results.json` | `b3_waiting_strict.py`、`strict_eval.py`（`verify_strict_eval.py` 为单元测试）；`verify_b3_waiting_strict.py` |
| 正文 Sec. III-C：预测到决策、单探针校准 | `predict_decide_results.json`、`waiting_budget_results.json` | `b3_predict_decide.py`、`b3_waiting_budget.py`；对应 `verify_*.py` |
| 正文 Sec. IV；补充 S1、S8，表 S2、S4、S5：数据角色、评分、协议追溯、任务差异 | `fgf2_expanded/expanded_results.json`；`audit/protocol_trace.json`；`audit/waiting_design_audit.json` 的 `task_differences` | `audit_protocols.py`、`audit_waiting_design.py` |
| 正文表 I、Sec. V-A；补充表 S3、图 S1(b)、S2 | `nested_results.json`、`nested_arx_results.json`、`nested_extensions_results.json` | `scripts/build_paper_assets.py` 生成 `core_rmse.tex`、`lopo_core.tex`、`prediction_phases.*` |
| 正文 Sec. V-B；补充图 S1(a)、S3：下一命令窗口 | `nested_results.json` 的 `phases`；`phase_detail.json`；`onset_split_results.json` | `nested_onset_split.py` |
| 正文表 II、Sec. V-C；补充 S8（孪生审计）、S9、表 S6–S8：认证校准下的等待（S7 为逐历史结果） | `waiting_strict_results.json`（`repeats/0/0.005/arms/certified`、`summary`、`statements` E1-S1–S4）；`audit/waiting_design_audit.json` | 生成 `strict_waiting.tex`、`strict_history.tex`、`strict_arms.tex`；旧表 `waiting_per_history.tex` 移到表 S6 |
| 正文表 III、Sec. V-D；补充 S11：预测误差与决策 | `predict_decide_results.json`（`summary`、`statements` E3-S1–S3） | 生成 `predict_decide.tex` |
| 正文图 2、Sec. V-E；补充 S10、表 S9：单探针校准预算 | `waiting_budget_results.json`（`summary`、`statements` E2-S1–S3） | 生成 `probe_budget.*`、`probe_budget.tex` |
| 正文图 3(a)、Sec. V-F；补充 S6：自然探测、后验预测检查、权重塌缩 | `natural_probe_results.json`、`recovery_check_results.json` | `natural_probe.py`、`b3_recovery_check.py` |
| 正文图 3(b)、Sec. V-G（"The Cost of Speed: Slack and Model Error"）；补充 S7、S12、表 S10、S11：模型失配与任务定义 | `waiting_mismatch_results.json`（图 3(b)，旧校准）；`waiting_scope_results.json`（E4-S1–S3，含锚定变体）；`window_tables.npz` | `b3_waiting_mismatch.py`、`b3_window_tables.py`、`b3_waiting_scope.py`；生成 `model_mismatch.*`、`scope_task.tex`、`scope_mismatch.tex` |
| 补充 S4、S5、图 S2：旧评价流程的前沿、延迟与条件图 | `waiting_frontier_results.json`、`delay_map_target_results.json` | 旧条件图（原正文图 2）移到补充图 S2(d) |
| 正文 Sec. V-H、VI：启示、证据边界、结论；补充 S13：复现入口 | — | — |

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

固定参照在 `reproduction/reference_metrics.json`：原 69 项冻结值保持不变，新增条目及原因写在 `extensions` 字段中，共 48039 项。`pipeline_checks.py` 检查正文中所有四位以上小数都在固定参照中，并检查正文引用的每个表、图都存在。`reproduction/run_report.json` 记录本次执行状态和环境，日志位于同目录的 `logs/`。

`release_manifest.json` 和 `simulation/checksums.json` 描述打包时的文件。所有生成文件都写成 LF，`.gitattributes` 保证在任何平台上检出的文本文件与清单逐字节一致。在不同库版本下重跑，结果的最后几位浮点数可能变化，论文数值以固定参照比较为准。
