# 补强清单逐项落实说明（2026-10-07）

对应表格：`revision/receiver_reuse_revision_plan.xlsx`（原表 15 项，已回填"状态""负责人""截止日期""备注"四列，"进度汇总"页的公式保持不变）。

原则与原表一致：所有新增分析先冻结方案、再只运行一次；原有结果全部保留，新结果作为新增冻结分析报告。四项新分析的方案、冻结记录、结果与核查报告如下，`python scripts/reproduce.py --mode verify` 会全部核查，并与扩展后的固定参照（14165 项）逐项比较。

| 新分析 | 方案 | 冻结记录 | 结果 | 核查 |
|---|---|---|---|---|
| 逐历史约束、oracle 差距、前沿、κ 与 $T_{\rm f}$ | `simulation/configs/b3_waiting_frontier_protocol.json` | `simulation/results/b3/b3_waiting_frontier_protocol_freeze.json` | `waiting_frontier_results.json` | `verify_b3_waiting_frontier.py` |
| 延迟 d 条件图 | `simulation/configs/b3_delay_map_protocol.json` | `simulation/results/b3/b3_delay_map_protocol_freeze.json` | `delay_map_results.json` | `verify_b3_delay_map.py` |
| 长时域预测、噪声地板、bootstrap、相位门控 | `simulation/configs/nested_extensions_protocol.json` | `simulation/results/nested/nested_extensions_protocol_freeze.json` | `nested_extensions_results.json` | `verify_nested_extensions.py` |
| 真实数据自然探测 | `simulation/configs/natural_probe_protocol.json` | `simulation/results/b3/natural_probe_protocol_freeze.json` | `natural_probe_results.json` | `verify_natural_probe.py` |

## P0（只需改写）

**1. 结论前后一致 — 已完成。** 摘要和第三条贡献写入负面结果：机理增量在下一命令窗口使误差升高 7.2%，并用自然探测解释原因（B3 高估长命令后的恢复）。增益全部量化：10 分钟误差降低 9.1% 与 4.5%（扣除噪声地板后 18.8% 与 9.8%），反馈等待弥合 42–45% 的 oracle 差距（7–8 分钟）。删去 "unified view"，结论只陈述有证据支持的部分。

**2. 符号与命名 — 已完成。** τ 只表示等待时间（滤波尺度改为 ℓ）；失败惩罚 D 改为 $T_{\rm f}$，评分式中的条件集 D 改为 $\mathcal{K}$；输入滤波 $q_\tau$ 改为 $v_\ell$，与可靠性目标 q 不再冲突；报告量滤波改为 $\bar y_\ell$，与 naive 上升 $r^{\rm nv}$ 区分；局部浓度 $c_R$ 改为 $\tilde u$，与阈值 c 区分。预测器改用描述性名称（Persist、Current、Filter、Lag、Filter+B3、Lag+B3、F+B3+Input），与 Test A/B/C 不再混淆；新增 Table I 特征表。代码与结果文件的键不变，对应关系写在 Table I 标题、`paper/supplementary_details.md` 与 `Paper_Simulation_Map_ZH.md`。

**3. 精简篇幅 — 已完成。** 归一化、滞后填充、B3 执行、校准细节、因果判据、AR(1) 白化公式等移到 `paper/supplementary_details.md`；原 Table IV（敏感性）移出正文，以一句话概括；散落的限定语集中到 V-G "Evidence Boundaries"。新增内容之后，论文从 7 页（上一版实际编译页数）缩到 6 页（含参考文献）。

**4. MC 文献定位 — 已完成。** 引言第二段新增并对比：带记忆的接收机（Kilinc & Akan 2013 序列检测与均衡；Mosayebi et al. 2014 利用记忆与采样率的译码器）、保护间隔/检测区间优化（Tepekule et al. 2015；Cao et al. 2020）、序贯检测与最优停止（Wald 1945；Tung & Mitra 2018 的 SPRT 收发机；Poor & Hadjiliadis 2009）。区别在于：这些工作的记忆在信道中、保护间隔事先固定；细胞接收机的记忆是只能经带噪报告量观察的内部状态，何时发送是一个停止问题。新文献均经检索核对，其中 3 篇未附 DOI（未能核实的 DOI 不写）。

## P1（现有数据新分析）

**5. 决策部分新指标 — 已完成。**
- oracle 差距：原合并约束下历史规则与平滑规则弥合了固定等待（按历史）到 oracle 差距的 33% 与 38%（主划分，即表中的 33–38%）。
- 逐历史约束：每条规则（含固定等待）对每种前次命令各取一个参数，要求每种历史都达标。目标 0.90 时所有规则只在 7–11/20 次划分中逐历史达标；0.94 时全部 20/20。此时反馈规则比逐历史固定等待（94/110 分钟）快 7.2–7.7 分钟，弥合 42–45%；当前观测规则不弥合。
- 主图：新 Fig. 3(a)(b) 为两种历史下的可靠性—时延前沿（评价集描述性扫描，标出 0.94 的校准工作点）；新 Table IV 给出逐历史结果与 G。

**6. 预测部分新指标 — 已完成。** 预测时域加长到 20、30 分钟；用稳健二阶差分估计白噪声地板（0.006–0.009），报告噪声调整 skill（Table III(c)）；决策相关指标采用"半衰减穿越"的平衡准确率（v2 的半衰减规则正是在该穿越时刻探测，预判 h 分钟后的穿越就是延迟为 h 的等待规则所需）。结果：Filter+B3 的 skill 在 10 分钟为 15.5%，20/30 分钟为 21%；显式输入历史在长时域失稳；机理增量提高穿越预判（但所有预测器都不可靠，平衡准确率 ≤ 67%）。

**7. 统计与设定依据 — 已完成。** 预测结果加按细胞分层的 bootstrap 区间（2000 次；正文注明细胞同批次，区间只反映细胞抽样）；oracle 成功率随 κ 变化（0.30–0.70）写入 Sec. II：κ = 0.5 是 oracle 仍 > 99% 的最大取值（99.25%；0.55 时 95.1%，0.6 时 82.0%），完整曲线在 `waiting_frontier_results.json` 的 `kappa_feasibility`（因篇幅只在正文给数值，未单独作图）；$T_{\rm f}=140$ 分钟为任何成功的最晚完成时间，160–260 分钟的敏感性显示增益 5.9–8.7 分钟、规则排序不变。

## P2（新仿真/新方法）

**8. 延迟 d 与条件图 — 已完成。** 引入延迟 d（反馈延迟加命令递送延迟），新增 B3-predictive 规则；对照组为"平滑 + 斜率"（平滑水平与最近增量的二维核，协方差取平滑白噪声的平稳协方差）。扫描 d（0–40 分钟）× 采样间隔（2/6/10 分钟）× 噪声（0.0025/0.005/0.01），20 次划分。结论：忽略延迟的 nowcast 从 d = 20 分钟起失效；预测型规则相对固定等待的增益随 d 从 5.5–5.9 降到 2.2–2.6 分钟；B3 模型在高噪声时每格都更快（0.9–5.5 分钟），稀疏采样时也有帮助，低噪声、密集采样时"平滑 + 斜率"更快。这把预测部分（h 步预测）与决策部分（等待）连接起来，回答了"何时需要机理模型"。

**9. 修补下一命令窗口 — 已完成（负面结果）。** 按方案实现两种只用训练协议拟合的变体：按阶段切换（训练数据中下一命令行上 Filter+B3 更好，切换被否决，结果等于 Filter+B3）和阶段交互权重（下一命令窗口误差 0.03571，反而高于 Filter+B3 的 0.03544；留出混合协议一折 0.03630）。原因：训练协议（持续刺激、3/20 短脉冲）没有"长命令后再探测"的情形，无法学到这一修正；自然探测（第 10 项）定位了 B3 的具体偏差。正文如实报告为未修复。

**10. 真实数据决策验证 — 已完成（部分验证）。** 25 ng/ml 下以 3/20 协议后续脉冲（以细胞自身首个脉冲为 naive）和混合协议第三个脉冲（以单次 5 分钟脉冲细胞为 naive）作为自然探测。固定的探测时刻不能评价完成时间，因此评价的是就绪判定：两处 B3 都预测无一成功，估计器没有作出"就绪"判定；实测 3/20 首个探测 19% 达标（被漏判）、之后 3–9%，混合协议 0/84。B3 高估长命令后的恢复（31% 对 4%），并低估 3/20 首个探测的上升（8% 对 16%）；B3 权重在实测细胞上坍缩，真实数据上的滤波需要过程噪声或细胞异质性。

## P3（留给期刊版）— 暂缓

11–15 项（conformal 风险控制、POMDP/最优停止联合优化、多细胞接收机、模型错配检验、前瞻实验）本轮未实现，表中标为"暂缓"。其中第 14 项需要其他候选网络结构（如 B1/B2）的模型文件与后验，作者仓库在固定提交中只提供 B3；第 15 项需要湿实验，验证方案见 `AI_MC_Complete_Validation_Plan_ZH.md` 与 `validation_kit/`。

## 仍需作者确认

- 目标会议的页数规定（本稿 6 页含参考文献）。
- 新增文献中 Kilinc & Akan 2013、Cao et al. 2020、Tung & Mitra 2018 的 DOI 与页码，建议投稿前在 IEEE Xplore 再核对一次。
- 主表选用校准目标 0.94（三个预设目标中所有规则逐历史 20/20 达标的最小值）是在看到结果后按这一规则选的，正文同时报告 0.90 和 0.92 的结果。
