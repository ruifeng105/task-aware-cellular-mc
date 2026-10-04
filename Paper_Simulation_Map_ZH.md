# 论文与仿真、数据和结果的对应

本表对应随包附带的五页初稿。所有路径相对于项目根目录。

| 论文内容 | 数据或结果 | 对应程序与状态 |
|---|---|---|
| Sec. IV：历史感知预测 | `simulation/results/fgf2_pilot/forecasts_*.csv`；`calibration_results.json` 的 `forecasting` | `simulation/code/calibrate_fgf2.py` 的 `forecasting_rows`、`model_features`、`forecasting`；岭回归在 `history_baselines.py`。已执行 |
| Table I：数据与划分 | `simulation/results/fgf2_expanded/expanded_results.json` 的 `inventory` | `expanded_fgf2.py` 的 `load_blocks`（预先登记的数据规则）；表由 `scripts/build_paper_assets.py` 生成 |
| Table II：试点 RMSE（含验证集与时间范围拆分） | `calibration_results.json` 的 `forecasting`（`validation_condition_equal_mse`、`test.time_support`） | `calibrate_fgf2.py` 的 `forecasting`、`time_support_metric`；`verify_fgf2_pipeline.py` 从保存预测重算评分 |
| Table III：预先登记的新方案检验 | `expanded_results.json` 的 `horizons`、`hypotheses` | `expanded_fgf2.py`（方案：`simulation/configs/fgf2_expanded_protocol.json`；冻结记录：`results/fgf2_expanded/protocol_freeze.json`）；`verify_expanded_fgf2.py` 从保存系数重算 70 个评分 |
| Table IV：留一方案检验 | `expanded_results.json` 的 `leave_one_protocol_out` | `expanded_fgf2.py` 的 `leave_one_protocol_out`；惩罚系数固定为主分析的选择 |
| 60 分钟单脉冲时序敏感性 | `expanded_results.json` 的 `sp60_timing_sensitivity` | 时序推断由 `verify_expanded_fgf2.py` 的 `sp60_timing_check` 从作者 B3 仿真复核 |
| Table V、Fig. 3：开环模型与 B3 参照 | `input_output_test_curves.csv`；`calibration_results.json` 的 `common_window_open_loop_comparison`；作者 `sim_post_mixed_*` 导出 | `calibrate_fgf2.py` 的 `input_output_identification`（含训练均值常数）与 `common_window_reference`；B3 仅重用作者导出 |
| 探索性历史匹配（正文 Sec. VII-E） | `exploratory_matched_history_pairs.csv`；`calibration_results.json` 的 `exploratory_history_matching` | `calibrate_fgf2.py` 的 `matched_history_pilot`；探索性分析，支持稀疏 |
| 来源、时序与因果性核查 | `fgf2_source_manifest.json`（47 个）、`fgf2_expanded_source_manifest.json`（56 个）；两份 `verification_results.json` | `verify_fgf2_pipeline.py`、`verify_expanded_fgf2.py`；可选的 `fetch_expanded_sources.py` 重新下载并核验 |
| Fig. 1：系统解释 | `paper/main.tex` 内的系统图 | 化学命令与细胞接收器的建模解释；局部浓度、输运和潜在状态尚未标定 |
| Algorithm 1（Fig. 2）：任务控制 | `simulation/configs/prospective_receiver_reuse.json` | 方法设计，`proposed_not_executed`；没有控制器代码或策略效益结果 |
| 相关工作与新颖性 | `paper/references.bib`；`simulation/novelty_audit.json`；中文核查报告 | 18 项定向核查及显式全文缺口，不是已完成的优先权证明 |

`paper/figures/forecast_mean_10min.*` 仍由脚本生成，作为补充图，正文不再引用。

## 使用统一入口

`python scripts/reproduce.py --mode verify` 核查保存的预测、扩展检验和固定论文数值；`--mode full` 从随包数据重跑试点与扩展检验，再核查并重建图表。两种模式都不增加训练数据、不改变划分、不重跑 B3 原始仿真器，也不执行任务控制。

固定参照在 `reproduction/reference_metrics.json`：原 69 项冻结值保持不变，新增条目（试点验证集与时间范围拆分、扩展检验全部指标、洗脱均值与计数等）及原因写在 `extensions` 字段中，共 416 项。`pipeline_checks.py` 还会检查正文中所有四位以上小数都在固定参照中。`reproduction/run_report.json` 记录本次执行状态和环境，日志位于同目录的 `logs/`。

`release_manifest.json` 和 `simulation/checksums.json` 描述打包时的文件。所有生成文件都写成 LF，`.gitattributes` 保证在任何平台上检出的文本文件与清单逐字节一致。在不同库版本下重跑，结果的最后几位浮点数可能变化，论文数值以固定参照比较为准。
