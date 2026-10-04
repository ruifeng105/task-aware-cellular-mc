# 论文与仿真、数据和结果的对应

本表对应随包附带的七页初稿。所有路径相对于项目根目录。

| 论文内容 | 数据或结果 | 对应程序与状态 |
|---|---|---|
| Algorithm 1：历史预测 | `simulation/results/fgf2_pilot/forecasts_2min.csv`、`forecasts_10min.csv`；`calibration_results.json` 的 `forecasting` | `simulation/code/calibrate_fgf2.py` 的 `forecasting_rows`、`forecasting`；岭回归实现位于 `history_baselines.py`。已执行 |
| Table I：数据与划分 | `calibration_results.json` 的 `inventory`、`split`；`simulation/configs/fgf2_condition_split.json` | `calibrate_fgf2.py` 的 `read_blocks`、`validate_split`；表由 `scripts/build_paper_assets.py` 生成 |
| Table II：个体预测 RMSE | 两个 `forecasts_*.csv` 与 `calibration_results.json` 的 `forecasting` | `verify_fgf2_pipeline.py` 从保存预测重算评分；按条件等权，非独立实验日统计 |
| Fig. 2：10 分钟预测均值 | `forecasts_10min.csv` | `build_paper_assets.py` 按目标时间聚合；图为均值曲线，Table II 评分仍基于个体误差 |
| Fig. 3：混合命令、测量与开环响应 | `input_output_test_curves.csv`；作者 B3 的 `sim_post_mixed_*_measurements.txt` 和 `sim_post_times.txt` | 简单模型由 `calibrate_fgf2.py` 拟合和计算；B3 仅重用作者导出；图由 `build_paper_assets.py` 生成 |
| Table III：0–180 分钟开环均值 RMSE | `calibration_results.json` 的 `common_window_open_loop_comparison` | `calibrate_fgf2.py` 的 `common_window_reference`；B3 有结构选择暴露，不是新盲测 |
| Table IV：探索性历史匹配 | `exploratory_matched_history_pairs.csv`；`calibration_results.json` 的 `exploratory_history_matching` | `calibrate_fgf2.py` 的 `matched_history_pilot`；探索性分析，支持稀疏，不能识别新的因果机制 |
| 来源、输入时序与因果性核查 | `simulation/public_data/fgf2_source_manifest.json`；`verification_results.json` | `verify_fgf2_pipeline.py`；47 个 Git blob、XML 命令、归一化和特征前缀检查 |
| Fig. 1：系统解释 | `paper/main.tex` 内的系统图 | 化学命令与细胞接收器的建模解释；局部浓度/输运和潜在状态尚未标定 |
| Algorithm 2：任务控制 | `simulation/configs/prospective_receiver_reuse.json` | 方法设计，`proposed_not_executed`；没有控制器代码或策略效益结果 |
| 相关工作与新颖性 | `paper/references.bib`；`simulation/novelty_audit.json`；中文核查报告 | 18 项定向核查及显式全文缺口，不是已完成的优先权证明 |

## 使用统一入口

`python scripts/reproduce.py --mode verify` 核查已保存的预测和固定论文数值。`--mode full` 从随包数据重跑模型，再核查并重建图表；不增加训练数据、不改变划分、不重跑 B3 原始仿真器，也不执行任务控制。

固定数值参照在 `reproduction/reference_metrics.json`，包含 12 个预测 RMSE、6 个开环 RMSE、4 个匹配均值以及轨迹/观测/匹配数量。`reproduction/run_report.json` 记录本次执行状态和环境。运行日志位于同目录的 `logs/`。

`release_manifest.json` 和 `simulation/checksums.json` 描述打包时的文件。重跑会更新结果、图表及日志，因此重跑后的文件不一定仍与打包清单逐字节相同；来源作者文件仍应通过 Git blob 核查，论文数值应通过固定参照比较。
