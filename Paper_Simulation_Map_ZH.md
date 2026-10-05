# 论文与仿真、数据和结果的对应

本表对应随包附带的六页初稿。所有路径相对于项目根目录。

| 论文内容 | 数据或结果 | 对应程序与状态 |
|---|---|---|
| Sec. III：历史感知预测 | `simulation/results/fgf2_pilot/forecasts_*.csv`；`calibration_results.json` 的 `forecasting` | `simulation/code/calibrate_fgf2.py` 的 `forecasting_rows`、`model_features`、`forecasting`；岭回归在 `history_baselines.py`。已执行 |
| Sec. III 式 (6)：B3 同反馈预测 M2 与嵌套残差 M3 | `simulation/results/b3/forecast_results.json` | `b3_forecast.py`（方案 `simulation/configs/b3_forecast_protocol.json`，冻结记录 `results/b3/b3_forecast_protocol_freeze.json`）；`verify_b3_forecast.py` 核查权重只用过去观测，并重算 60 个评分 |
| Sec. V-C：B3 执行与验收 | `simulation/results/b3/verification_results.json` | `b3_model.py` 解析作者模型、初值与测量文件，参数顺序取自作者仿真摘要；`verify_b3.py` 检查来源、结构、守恒律、无刺激稳态和复现门槛；来源清单 `fgf2_b3_source_manifest.json`，可选 `fetch_b3_sources.py` |
| Table I：数据与划分 | `simulation/results/fgf2_expanded/expanded_results.json` 的 `inventory` | `expanded_fgf2.py` 的 `load_blocks`（预先登记的数据规则）；表由 `scripts/build_paper_assets.py` 生成 |
| Table II：试点 RMSE（含验证集与时间范围拆分） | `calibration_results.json` 的 `forecasting`（`validation_condition_equal_mse`、`test.time_support`） | `calibrate_fgf2.py` 的 `forecasting`、`time_support_metric`；`verify_fgf2_pipeline.py` 从保存预测重算评分 |
| Table III：新方案检验（含 M2、M3 两行） | `expanded_results.json` 的 `horizons`、`hypotheses`；M2、M3 来自 `results/b3/forecast_results.json` | `expanded_fgf2.py`（方案：`simulation/configs/fgf2_expanded_protocol.json`；冻结记录：`results/fgf2_expanded/protocol_freeze.json`）；`verify_expanded_fgf2.py` 从保存系数重算 70 个评分 |
| Table IV：留一方案检验 | `expanded_results.json` 的 `leave_one_protocol_out` | `expanded_fgf2.py` 的 `leave_one_protocol_out`；惩罚系数固定为主分析的选择 |
| 60 分钟单脉冲时序敏感性 | `expanded_results.json` 的 `sp60_timing_sensitivity` | 时序推断由 `verify_expanded_fgf2.py` 的 `sp60_timing_check` 从作者 B3 仿真复核；作者仿真摘要 `sim_post_model_summary.txt` 明确写为 `[0, 60)` |
| Sec. VI-E、Fig. 3：B3 上的仿真准备度与等待策略 | `simulation/results/b3/readiness_results.json`、`readiness_decisions.csv`、`readiness_tables.npz`；第 1 版结果 `readiness_results_v1.json` | `b3_readiness.py`（方案第 2 版 `b3_readiness_protocol_v2.json`，第 1 版 `b3_readiness_protocol.json`）；`verify_b3_readiness.py` 抽样重新仿真、复核第 1 版不可行，并重算 16000 个决策 |
| Table V：开环模型与 B3 参照 | `input_output_test_curves.csv`；`calibration_results.json` 的 `common_window_open_loop_comparison`；作者 `sim_post_mixed_*` 导出 | `calibrate_fgf2.py` 的 `input_output_identification`（含训练均值常数）与 `common_window_reference` |
| 探索性历史匹配（正文 Sec. VI-G） | `exploratory_matched_history_pairs.csv`；`calibration_results.json` 的 `exploratory_history_matching` | `calibrate_fgf2.py` 的 `matched_history_pilot`；探索性分析，支持稀疏 |
| 来源、时序与因果性核查 | `fgf2_source_manifest.json`（47 个）、`fgf2_expanded_source_manifest.json`（56 个）、`fgf2_b3_source_manifest.json`（3 个）；各 `verification_results.json` | `verify_fgf2_pipeline.py`、`verify_expanded_fgf2.py`、`verify_b3.py`；可选的 `fetch_*_sources.py` 重新下载并核验 |
| Fig. 1：系统解释 | `paper/main.tex` 内的系统图 | 化学命令与细胞接收器的建模解释；局部浓度、输运和潜在状态尚未标定 |
| Sec. IV 式 (7)：准备度感知的复用控制 | `simulation/configs/prospective_receiver_reuse.json`、`validation_kit/configs/validation_registration_draft.json` | 真实接收器上仍是方法设计；只在 B3 上做了等待策略仿真。原 Algorithm 1 文字框已改为正文中的滚动规划说明 |
| Fig. 2：单个接收器上的准备度等待示例 | `simulation/results/b3/readiness_example.json` | `b3_readiness_example.py` 按固定规则从保存的决策中选样本（30 分钟历史；"仅当前观测"策略过早探测失败、历史策略成功的 194 个样本中，历史策略等待时间取中位数者），复算观测、两种成功概率估计和三个等待时刻的探测响应；`verify_b3_readiness.py` 核对选样规则、决策与保存表格一致 |
| Sec. VII：前瞻验证设计 | `AI_MC_Complete_Validation_Plan_ZH.md`；`validation_kit/` | `validation_kit/scripts/` 生成候选设计与样本规划并审计登记；登记状态为 `draft_not_ready`，没有实验记录 |
| 相关工作与新颖性 | `paper/references.bib`（14 篇）；`simulation/novelty_audit.json`；中文核查报告 | 18 项定向核查及显式全文缺口；新增 3 篇分子通信文献经 Crossref 核对，只按摘要层面引用 |

`paper/figures/mixed_response.*` 和 `forecast_mean_10min.*` 仍由脚本生成，作为补充图，正文不再引用。

## 使用统一入口

`python scripts/reproduce.py --mode verify` 核查保存的预测、扩展检验、B3 执行与两项 B3 分析，以及固定论文数值（需要 SciPy，约 3 分钟）。`--mode full` 从随包数据重跑试点、扩展检验、B3 预测与准备度仿真（准备度仿真约 5 分钟），再核查并重建图表与规划工具输出。两种模式都不增加训练数据、不改变划分、不重新推断 B3 参数，也不在真实接收器上执行任务控制。

固定参照在 `reproduction/reference_metrics.json`：原 69 项冻结值保持不变，新增条目（试点验证集与时间范围拆分、扩展检验、B3 验收、M0–M3、准备度研究与图 2 示例等）及原因写在 `extensions` 字段中，共 561 项。`pipeline_checks.py` 还会检查正文中所有四位以上小数都在固定参照中。`reproduction/run_report.json` 记录本次执行状态和环境，日志位于同目录的 `logs/`。

`release_manifest.json` 和 `simulation/checksums.json` 描述打包时的文件。所有生成文件都写成 LF，`.gitattributes` 保证在任何平台上检出的文本文件与清单逐字节一致。在不同库版本下重跑，结果的最后几位浮点数可能变化，论文数值以固定参照比较为准。
