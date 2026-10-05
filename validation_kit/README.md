# AI–MC 完整验证方案配套工具

创建日期：2026-10-04。完整研究方案位于压缩包根目录 `AI_MC_Complete_Validation_Plan_ZH.md`。

这是前瞻验证的规划工具包。本包新增真实实验会话数为 **0**，所有 `data_templates` CSV 都只有表头。已有历史项目的源模型/论文/仿真位于原项目归档，本包不重复包含。新粒子滤波、历史残差、概率标定、任务标签器和闭环控制器尚未实现。

## 1. 快速使用

使用 Python 3.12 或兼容环境。依赖写在 `requirements.txt`；实际校验环境另存于 `planning_outputs/runtime_environment.json`。本包没有执行任何硬件命令。

在 `validation_kit` 目录执行：

```bash
python scripts/design_candidates.py --stage pilot
python scripts/design_candidates.py --stage main
python scripts/sample_size.py
python scripts/audit_registration.py
python scripts/audit_data_contract.py
python scripts/verify_planning_tools.py
```

脚本默认路径根据脚本位置解析，因此也可从其他目录用绝对路径调用。每项输出写入 `planning_outputs`。实际记录填好后可指定新 CSV，保持空白模板原样：

```bash
python scripts/audit_data_contract.py \
  --manifest /absolute/path/to/actual_session_manifest.csv \
  --observations /absolute/path/to/actual_observations.csv \
  --output /absolute/path/to/data_contract_audit.json
```

正式冻结前使用：

```bash
python scripts/audit_registration.py --config /absolute/path/to/frozen_registration.json --require-ready
```

当前草案会返回退出码 **2**，并报告缺失字段；这是正确结果。该检查用于正式模型/控制验证的登记完整性，不能把它当作生物平台鉴定、实验审批或执行器互锁。P1/P2 开发采集不要求先训练完成所有模型，但其实际平台、输入、记录与实验协议仍须事先明确。

## 2. 目录与功能

| 文件 | 功能 | 本轮输出 |
|---|---|---|
| `configs/validation_registration_draft.json` | 候选设计、待填任务/组/测量/统计冻结字段 | 草案，不是已注册协议 |
| `scripts/design_candidates.py` | 同钟历史—探测及匹配控制清单 | 预试 10、主设计 54 候选变体 |
| `scripts/sample_size.py` | 精确成功率下界和配对日期成本规划 | 规划示例，不是实测可靠性 |
| `scripts/audit_registration.py` | 任务、组、动作与测量声明的结构检查 | `draft_not_ready` |
| `scripts/audit_data_contract.py` | 实际 ID、组泄漏、反馈可用时间与缺失规则 | 空数据为 `awaiting_actual_records` |
| `scripts/verify_planning_tools.py` | 关键解析/软件行为检查 | `passed`，无生物验证 |
| `data_templates/*.csv` | 真实采集/全文证据的空白接口 | 7 张只有表头的文件 |

候选 CSV 中的 `provisional_candidate_order` 只是固定种子的置换。真实分块随机化必须在实验前按实际设备能力登记。`actual_day_id` 等列为空，不能将 `pilot_C001` 等计划条件 ID 伪装成实际实验日或会话。

54=两个时钟块×每块 27 条件，不是 54 个独立试验。预刺激/探测浓度、时长、等待和时钟均未标定。若修改设计因子，需重新核对条件覆盖、生成清单，并相应更新校验脚本中固定的 10/54 设计假定。

## 3. 登记字段的填写顺序

先填实际平台、时间同步、流量/FGF2 测量等级与原始来源；再用独立应用标定固定任务/资格/掉踪规则；取得先导组级变异后确定最小有意义差、样本量与统计程序；训练和标定完成后冻结模型、动作与回退。正式测试前保存 `evidence.freeze_record` 对应版本/哈希和日期。

`group_split` 四组为训练、模型选择、概率标定、锁定测试；锁定测试包含互不共享实际日期/最大共享单位的 `prediction_validation` 与 `prospective_control`。二者组 ID 的并集必须等于 `locked_test` 列表。已看过旧测试的数据只作开发。

任务对象列表至少两项，各项需要 `response_lower`、`response_upper`、`hold_min`、`deadline_min`、`minimum_native_frames`，且须在完整协议中另写确认窗口、假激活和原生最大间隔处理。资格细胞集合在整个序列起点固定。`sequence_reliability_target` 必须在 0 与 1 之间；有限实验不能证明绝对成功概率为 1。

普通审计不会证明原始文件存在、仪器准确、组真正独立、标定充分或正文已经核查。即使字段齐全，状态也仅为 `registration_fields_complete_pending_evidence_review`。`evidence.nearest_prior_fulltext_comparison_completed` 在本草案为 false，投稿前仍需完整查新。

## 4. 七张记录接口与单位

### `session_manifest.csv`

一行对应一个真实会话，实际 `session_id` 唯一。日期、腔室、细胞/器件批次、最大共享 `split_group_id` 均来自采集记录。`partition` 为 `pilot`、`train`、`model_selection`、`probability_calibration` 或 `locked_test`；测试行的 `test_cohort` 必填 `prediction_validation` 或 `prospective_control`，其余行留空。相同 day/group 不可跨分区，也不可跨测试子队列。

`eligible_cell_count_before_command` 是序列资格判定的预先数量；存在多个序列时，以各序列 outcome 及细胞资格记录为准。`raw_data_path`/`raw_data_sha256` 保存来源与校验和。结构检查脚本不读取/验证原始文件内容，应另核对。

### `observations.csv`

一行一个原生细胞读出点，包含会话、序列、细胞 ID。`time_min` 是相对会话共同时间零点的采集分钟；`arrival_time_min` 是控制端实际可用分钟，必须不早于采集，须先实测时钟同步。原始 FRET 比率和因果基线归一化值分开记录。

缺失响应的 `response_missing=1`，`normalized_fret` 留空，填写真实 `missing_reason`；原始通道若还能取得可保留，不能补零。`eligible_at_sequence_start` 固定为序列开始的资格标志，不随之后表现变化。审计不生成任务标签，也不把现有细胞变成独立实验单位。

### `delivery_commands.csv`

保存实际会话/day/chamber、执行时间和浓度命令，单位 min、ng/ml。与旧项目命令接口列名一致。实际候选与执行不同的情况还需在控制日志登记；生成的规划 CSV 不能直接替代这张表。

### `delivery_samples.csv`

入口/细胞位置 FGF2：ng/ml；流量：ml/min；采集与到达：min；示踪归一化值单列。缺少通道留空。示踪不能填入 FGF2 浓度列。离线检测值的到达时刻按实际获得时间登记，不能在在线训练/重放中提前使用。

FGF2 与流量采样可按真实异步时刻记录；物理积分需要另登记对齐、插值、覆盖与测量误差。本包的数据结构审计只检查 manifest/observations，不会审计这张表的化学数值或自动产生积分。

### `control_event_log.csv`

记录决策/下发/执行/确认/终止与回退。时间仍以共同零点表示；运行时间单位 min。`data_available_until_min` 是该次决策所用已到达数据的采集上限，不是未来窗口终点。回退仍归入最初 `assigned_policy`。

### `sequence_outcomes.csv`

一行对应所有已分配的真实任务序列，包括失败、掉踪和回退。`sequence_success` 由冻结任务规则计算；完成/截止为从共同序列起点起算的持续时间，单位 min，避免把会话绝对时钟误作持续时间。

`failure_penalized_sequence_completion_time_min`：成功用注册完成时间，失败用相同期限 D。真实资源分别记录：入口质量 ng、局部暴露 ng·min/ml、命令代理 ng·min/ml、缓冲液体积 ml。缺测积分留空，不填零。`joint_success_cells` 必须是同一资格队列完成全部任务的细胞数，不能用各任务边缘成功人数的乘积代替。

该表需要未来任务标签/统计模块或经核查的人工标注填入。本包不计算真实成功率、策略成本收益或因果效果。

### `novelty_evidence.csv`

逐篇记录主文/SI 版本与获取状态、实际链路/接收器、同信息参照、独立组、联合任务、实测成本、贡献重叠和证据位置。关键邻居没有正文时标为待取，不把摘要未写当作全文不存在。原 22 条查新记录仍在原项目，本表是追加核查接口。

## 5. 样本规划的适用条件

默认可靠性示例为单个主张、单侧 α=0.05、真正独立且全部成功：q=0.90/0.95/0.99 时至少需 29/59/299 次。它们不是实际次数或正式样本量。共享腔室的细胞、共享日期的相关序列不能直接当作独立二项观测。

成本规划采用已知配对日期差标准差的正态近似，输出 0.2/0.3/0.5/0.8/1.0 标准化效应示例。实际正式 N 必须结合先导方差、组相关、失败、可靠性非劣界和多重比较制定。工具支持错误预算变化，例如：

```bash
python scripts/sample_size.py --alpha 0.025 --power 0.8 --output /absolute/path/to/planning_output
```

此命令只改变示例计算；不自动完成正式功效分析或冻结协议。

## 6. 本轮校验边界

已检查：设计计数/同钟对齐/匹配控制，非法反馈时间、跨分区与测试子队列泄漏拒绝，以及精确下界和 SciPy 单侧接口一致。内部合成记录只存在于软件测试内存，均带 `SW_fixture` 标签，没有输出为生物数据。

未验证：实际仪器、培养/报告器、FGF2 通道、样本独立性、任务阈值、状态估计器、任何新的训练/模型收益、闭环控制以及最近邻全文穷尽性。研究结果要按完整方案 P0–P6 逐项获得。
