# 接收机复用评价修订：实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按用户方案（`C:\Users\ruife\OneDrive\桌面\cellular-receiver-revision-plan-zh.md`）完成 P0 核查、E1 严格可靠性评价、E2 有限探针校准、E3 预测到决策、E4 任务与失配稳健性，并把论文正文改写为不超过 7 页、结论用语统一。

**Architecture:** 每项新分析沿用仓库惯例：先写协议 JSON，再用 SHA-256 冻结记录，单次运行，最后由独立 verifier 复算并做单元测试（先 RED 后 GREEN）。所有仿真分析复用 `readiness_tables.npz`；只有 E4 的观察窗口变体需要新的 B3 探针仿真。新的公共工具（去重、三角色划分、Clopper–Pearson 下界、固定序列校准）集中在一个模块里，E1–E4 共用。

**Tech Stack:** Python 3（numpy、scipy、pandas、matplotlib），LaTeX（IEEEtran），Windows 下的 spawn 多进程（`OPENBLAS_NUM_THREADS=1`）。

**Spec:** `C:\Users\ruife\OneDrive\桌面\cellular-receiver-revision-plan-zh.md`（用户上传的修改方案；执行时与本计划一起读）。

## Global Constraints

- 只有用户明确要求时才 commit 或 push；commit 信息以 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` 结尾。
- 正文上限 7 页（用户自行压缩多余部分）；补充材料不限。
- 新分析流程：协议 → SHA-256 冻结记录 → 单次运行 → verifier（RED→GREEN）。如果协议是在探索之后才冻结，必须在冻结记录里写明。
- 所有输出文件只用 LF；不得覆盖用户改过的 `paper/figures/fig1_system_concept.pdf` 和 `paper/references.bib`。
- 不得结束其他项目或其他用户的进程，只能结束自己启动的进程。
- 用语：不用"保证 90%"、"至少 35.5 min"、"实测下界"、"最优停止规则"、"上界"（指完整曲线校准时）；κ=0.5 称为响应恢复代理任务。
- 所有结果无论方向如何都要报告；不可行和回退的情形必须进入分母。
- 每个成功率都要写明抽样单位：一个不重复的 B3 后验参数向量加一次独立的观测噪声抽样。它反映群体平均参数的不确定性，不代表细胞异质性。

## Review Focus

1. **后验重复样本造成自匹配**：`posterior.txt` 的 1000 行里只有 948 个不同向量（44 对、4 组三重）。按索引留一并不能排除"孪生"样本，B3 belief 会因此近乎完美地命中。预期行为：E1–E4 都按不同向量去重，审计中量化旧表 II 受影响的程度。测试放在 Task 1 和 Task 3。
2. **校准不可行时被静默丢弃**：没有任何参数通过认证时，规则必须回退为 120 min 探针，并计入成功率与完成时间。测试放在 Task 3（人工构造全部失败的合成情形）。
3. **测试结果泄漏进校准**：把测试接收机的结果取反后，所有已选参数必须不变。测试放在 Task 3、Task 4、Task 5。
4. **固定序列在第一步失败**：最保守的候选未通过时，可接受集合必须为空，而不是跳到后面的候选。测试放在 Task 3。
5. **有限探针下标签越界**：一个校准接收机只能贡献其被探测时刻的那一个结果；访问其他等待时刻的标签必须报错。测试放在 Task 4（用一个只暴露单个标签的访问器来实现）。

---

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `simulation/code/audit_protocols.py` | P0：追溯每个协议的刺激时间表（作者 `pulses.txt`、`normalize_data.m`、XML、`sim_post_model_summary.txt`），并与 `calibrate_fgf2.segments` 和截断原点逐项对应；检查截断原点之前没有刺激响应 |
| `simulation/code/audit_waiting_design.py` | P0：后验重复、样本角色、留一自匹配、不可行计数、实验与仿真任务差异表；把旧表 II（frontier，目标 0.94）按"排除有孪生样本的评价接收机"重新打分 |
| `simulation/code/strict_eval.py` | 公共工具：`distinct_samples`、`three_way_split`、`cp_lower`、`fixed_sequence`、`certified_choice`、`paired_bootstrap` |
| `simulation/code/b3_waiting_strict.py` + `configs/b3_waiting_strict_protocol.json` | E1 |
| `simulation/code/b3_waiting_budget.py` + `configs/b3_waiting_budget_protocol.json` | E2 |
| `simulation/code/b3_predict_decide.py` + `configs/b3_predict_decide_protocol.json` | E3 |
| `simulation/code/b3_window_tables.py` | E4：模拟 [w, w+30] 的探针响应，生成 W=10/20/30 的上升量表（W=20 必须复现现有表） |
| `simulation/code/b3_waiting_scope.py` + `configs/b3_waiting_scope_protocol.json` | E4：κ/W/D 单因素敏感性，以及幅度、恢复速度、接收机间差异三类失配 |
| `simulation/code/verify_*.py`（每个分析一个） | 冻结检查、单元测试、全量复算、读取规则复核 |
| `scripts/reproduce.py` | 新增步骤与 `revision8_metrics()` |
| `scripts/build_paper_assets.py` | 新表 II（E1）、新图 2（E2 预算曲线）、改正图 3 的等效延迟标注、补充材料表 |
| `paper/main.tex`、`paper/supplement.tex` | 重写贡献、摘要、II–VI 节；把条件热图移入补充材料 |
| `README.md`、`Paper_Notes_ZH.md`、`Paper_Simulation_Map_ZH.md`、`revision/Revision_Response_ZH.md` | 文档同步 |

---

### Task 1: P0 核查（协议追溯与等待设计审计）

**Files:**
- Create: `simulation/code/audit_protocols.py`
- Create: `simulation/code/audit_waiting_design.py`
- Create: `simulation/results/audit/protocol_trace.json`, `simulation/results/audit/waiting_design_audit.json`

**Interfaces:**
- Consumes: `calibrate_fgf2.segments(protocol)`, `b3_model.load_posterior()`, `b3_waiting.split_samples`, `b3_waiting_robustness.seeds`, `b3_waiting_frontier.candidates/choose/lookup`
- Produces: `strict_eval.distinct_samples(posterior) -> np.ndarray[int]`（每个不同向量第一次出现的索引，升序），供 Task 3–6 使用

- [ ] **Step 1：写协议追溯检查（先跑，确认在实现前失败）**

`audit_protocols.py` 的 `main()` 断言下列各项，并把逐项表写入 JSON：
  - `fgf_mixed/pulses.txt` 为 `43-46, 66-96, 156-161`（原始分钟）；`normalize_data.m` 的 `first_pulse_index = 22`，`time.txt` 第 22 个值（从 1 开始计）为 42 min，所以截断原点是 42；`pulses.txt − 42 == pulses_trunc.txt == segments('fgf_mixed') == [(1,4),(24,54),(114,119)]`；XML 的三个 `<input>`（startingtime 1/24/114，duration 3/30/5）与之相同；`sim_post_model_summary.txt` 的 mixed 实验行也相同。
  - 对每个剂量 `fgf_mixed/{dose}_unnormalized.txt`：重算 `unnormalized / median(t∈[10,30])`，从第 22 列开始应与 `{dose}_trunc.txt` 在 1e-6 以内相等。这证明截断文件来自原始记录，原点为 42 min。
  - 截断原点之前（原始 0–42 min）：每个细胞的归一化报告值相对 [10,30] min 中位数的最大偏离，取群体中位数，用来说明更早的刺激没有进入记录；输出数值，不预设阈值。
  - 3/20、sp_5、sp_10、sus：`time.txt` 与 `time_trunc.txt` 的长度差给出截断原点；XML 与 summary 的周期、时长、起点与 `segments` 一致；sp_60 没有 XML 输入，记为"由作者仿真导出推断"（沿用代码注释）。
  - 输出对照表：协议、脉冲起止（截断坐标与原始坐标）、时长、间隔、剂量、来源文件、时间单位（min）。

- [ ] **Step 2：写等待设计审计**

`audit_waiting_design.py` 输出：
  - 重复结构：不同向量数（948）、重数直方图（{1: 900, 2: 44, 3: 4}）。
  - 对 frontier 的 20 个划分：评价接收机在校准（参考）集合里有孪生样本的数目；校准接收机在留一参考集合里有孪生样本的数目。
  - 影响量化：在 noise 0.005、目标 0.94 下，用已保存的 choices 重新打分；分为全部评价接收机和排除有孪生的评价接收机两组，比较各规则的逐历史成功率与平均完成时间，以及 B3 belief 相对固定等待的配对差。仅作描述。
  - 不可行处理：从 `waiting_frontier_results.json` 汇总每个目标、噪声、规则下 `target_reached=False` 的划分数，并说明旧回退规则（"成功率最高的参数"，阈值通常为 0.99）。
  - 任务差异表（写成 JSON 文本字段，后面进补充材料）：探针时长（3/20 为 3 min，mixed 为 5 min，仿真为 5 min）；naive 分母（3/20 为同一细胞第一个脉冲；mixed 为另一组 sp_5 细胞的群体中位数；仿真为同一样本）；响应（实测为 3 帧滑动平均且有噪声，起点为前两帧均值；仿真无噪声，1 min 网格，起点为 F(w)）；窗口均为 20 min。结论：自然探针和恢复核查属于模型检验，只有仿真等待研究与式（2）完全一致。

- [ ] **Step 3：运行两个审计并检查输出**

Run: `cd simulation/code && python audit_protocols.py && python audit_waiting_design.py`
Expected: 两个 JSON 写出，所有断言通过；打印孪生样本计数和受影响的差值。

- [ ] **Step 4：把两个审计接入 `scripts/reproduce.py` 的 verify 步骤**（审计本身就是确定性检查，不需要冻结记录）。

---

### Task 2: 公共工具 `strict_eval.py`（先测试）

**Files:**
- Create: `simulation/code/strict_eval.py`
- Test: `simulation/code/verify_strict_eval.py`

**Interfaces（Produces）:**
```python
def distinct_samples(posterior: np.ndarray) -> np.ndarray          # 每个不同行第一次出现的索引，升序
def three_way_split(ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]  # R, C, T 各 len//3，升序，互不相交
def cp_lower(k: int, n: int, alpha: float) -> float                 # 单侧 (1-alpha) Clopper-Pearson 下界；k=0 时为 0
def binom_pvalue(k: int, n: int, q: float) -> float                 # P(Bin(n,q) >= k)
def fixed_sequence(successes: Sequence[tuple[int,int]], q: float, alpha: float) -> int   # 从头开始连续被拒绝的个数 m（可接受集合为前 m 个）
def certified_choice(ordered: list[tuple[dict, np.ndarray, np.ndarray]], q: float, alpha: float, chains: int = 1) -> dict
def paired_bootstrap(diff: np.ndarray, reps: int, seed: int) -> list[float]   # 均值的 95% 百分位区间
```

- [ ] **Step 1：写失败的单元测试**（`verify_strict_eval.py`）

```python
import numpy as np
from scipy import stats
import strict_eval as se

def test_cp_lower():
    for k, n in ((0, 10), (9, 10), (300, 316), (316, 316)):
        ref = 0. if k == 0 else stats.beta.ppf(.025, k, n - k + 1)
        assert abs(se.cp_lower(k, n, .025) - ref) < 1e-12
    # duality: lower bound >= q  <=>  P(Bin(n,q) >= k) <= alpha
    for k in range(280, 317):
        assert (se.cp_lower(k, 316, .025) >= .9) == (se.binom_pvalue(k, 316, .9) <= .025)

def test_fixed_sequence_stops_at_first_failure():
    assert se.fixed_sequence([(316, 316), (310, 316), (250, 316), (316, 316)], .9, .025) == 2
    assert se.fixed_sequence([(250, 316), (316, 316)], .9, .025) == 0     # first rung fails -> empty set

def test_split_disjoint_and_distinct():
    post = np.array([[1, 2], [1, 2], [3, 4], [5, 6], [7, 8], [9, 9], [1, 2]], float)
    ids = se.distinct_samples(post)
    assert ids.tolist() == [0, 2, 3, 4, 5]
    r, c, t = se.three_way_split(np.arange(948), seed=1)
    assert len(r) == len(c) == len(t) == 316 and not (set(r) & set(c) or set(r) & set(t) or set(c) & set(t))

def test_certified_choice_fallback():
    ok = np.zeros((2, 50), bool); comp = np.full((2, 50), 140.)
    choice = se.certified_choice([(dict(threshold=.99), ok, comp)], .9, .025)
    assert choice['certified'] is False and choice['fallback'] == 'probe_at_120'
```

- [ ] **Step 2：运行，确认失败**

Run: `cd simulation/code && python verify_strict_eval.py`
Expected: `ModuleNotFoundError: No module named 'strict_eval'`

- [ ] **Step 3：实现**

```python
"""Shared helpers for the strict waiting evaluations (E1-E4): distinct posterior vectors, three-way roles,
one-sided Clopper-Pearson bounds, fixed-sequence calibration and paired bootstrap intervals."""
import numpy as np
from scipy import stats

def distinct_samples(posterior):
    _, first = np.unique(posterior, axis=0, return_index=True)
    return np.sort(first)

def three_way_split(ids, seed):
    order = np.asarray(ids)[np.random.default_rng(seed).permutation(len(ids))]
    m = len(ids) // 3
    return np.sort(order[:m]), np.sort(order[m:2 * m]), np.sort(order[2 * m:3 * m])

def cp_lower(k, n, alpha):
    return 0. if k == 0 else float(stats.beta.ppf(alpha, k, n - k + 1))

def binom_pvalue(k, n, q):
    return float(stats.binom.sf(k - 1, n, q))

def fixed_sequence(successes, q, alpha):
    m = 0
    for k, n in successes:
        if binom_pvalue(k, n, q) > alpha:
            break
        m += 1
    return m

def certified_choice(ordered, q, alpha, chains=1, h=None):
    """`ordered`: list of chains, or one chain of (label, ok, completion) from most conservative to most
    aggressive; arrays (H, n), scored for history h. Each chain is tested at alpha / chains; among all admitted
    parameters the smallest calibration mean completion wins (ties: tie_key). No admitted parameter ->
    fallback probe at 120 min, certified False."""
```
（`certified_choice` 的完整实现按上面的注释写：对每条链调用 `fixed_sequence`，收集可接受参数，按 `(mean completion, tie_key)` 取最小；为空时返回 `dict(certified=False, fallback='probe_at_120')`；否则返回参数加 `certified=True` 和校准统计。`paired_bootstrap` 用 `np.random.default_rng(seed).integers` 抽取下标，取均值的 2.5/97.5 百分位。）

- [ ] **Step 4：运行，确认通过**

Run: `cd simulation/code && python verify_strict_eval.py`
Expected: `strict_eval helpers: passed`

---

### Task 3: E1 冻结规则后的严格可靠性评价

**Files:**
- Create: `simulation/configs/b3_waiting_strict_protocol.json`, `simulation/results/b3/b3_waiting_strict_protocol_freeze.json`
- Create: `simulation/code/b3_waiting_strict.py`, `simulation/code/verify_b3_waiting_strict.py`
- Output: `simulation/results/b3/waiting_strict_results.json`

**Interfaces:**
- Consumes: `strict_eval.*`、`b3_waiting.estimator_probabilities/noisy_observations`、`b3_waiting_frontier.candidates/crossing_index/scored/oracle_outcomes`、`b3_readiness.success_table/TABLES/GRID`
- Produces: `run_split(r, noise, arm) -> dict`、`TEST_ROLES(r) -> (R, C, T)`、`choose_arm(options_cal, arm) -> choices`，E2–E4 复用

**协议要点（冻结前写入 JSON）：**
- 总体：948 个不同后验向量（`distinct_samples`）；接收机等于一个向量加一次独立的观测噪声抽样（噪声种子 `20261109 + 100000·r`，按 plant 与 history 偏移）。
- 角色：每个划分 r（0–19，种子 `20261009 + r`）分成 R/C/T 各 316。估计器参考集只用 R（无需留一，角色不相交）；规则参数只在 C 上选；T 只用于最终评价。划分 0 为主结果，20 个划分只报告稳定性。
- 规则族与候选（与表 II 相同）：每种历史一个固定等待（0–120，步长 2）；current 阈值 0.50–0.99；smoothed τ∈{2,4,8,16} × 阈值；B3 belief 阈值。
- 三种校准臂：
  - `certified`（主）：每种历史按固定序列做单侧二项检验，H0 为 S_h(θ) < 0.9，α_h = 0.025，两种历史经 Bonferroni 合并后联合水平为 0.05。阈值从 0.99 递减到 0.50，固定等待从 120 递减到 0。smoothed 的 4 个 τ 各成一条链，每条用 α_h/4。在可接受集合中取校准平均完成时间最短的参数；集合为空时，该历史在 120 min 探针，标记 `certified=false`，结果计入统计。
  - `plugin_090`：校准成功率 ≥ 0.90 的参数中完成时间最短者（frontier 的旧规则，回退也相同）。
  - `plugin_094`：同上，目标为 0.94。标注：该目标是在旧研究看到评价结果之后才选定的。
- 噪声：0.005（主）、0.01（次）。
- T 上的指标（每臂、每规则、每种历史，外加 oracle）：n、成功数、成功率、单侧 95% 与 97.5%（联合）CP 下界、带失败惩罚的平均完成时间（D=140）、成功条件下的平均完成时间、平均等待、失败数、是否已认证；配对差（各自适应规则减同臂固定等待，逐接收机计算）的完成时间与成功率，附 95% 配对 bootstrap 区间（2000 次，种子 20261012）。
- 20 个划分的汇总：每臂每规则，T 上两种历史成功率都 ≥ 0.9 的划分数；两种历史 97.5% 下界都 ≥ 0.9 的划分数；校准认证成功的划分数；完成时间与配对差的中位数和范围。
- 预注册陈述：
  - E1-S1：certified 臂、划分 0、噪声 0.005 下，各规则 T 上逐历史成功率及下界。
  - E1-S2：同一设置下自适应规则减固定等待的配对完成时间差及其区间，并给出 20 个划分的中位数。
  - E1-S3：plugin_090 臂在多少个划分中满足逐历史 0.9。
  - E1-S4：certified 臂与 plugin_094 臂的完成时间差，即认证所需的代价。
- 解释边界：948 个向量都在开发阶段用过（规则族、网格和旧目标都是看过这些样本的结果后确定的）。因此 E1 覆盖的是划分抽样不确定性和观测噪声，不覆盖未见过的参数接收机，也不覆盖细胞异质性。

- [ ] **Step 1：写协议 JSON 和冻结记录**（`known_before_freezing` 写明已知旧表 II 数值、重复样本问题和 0.94 的来源；`statement` 写"在任何 E1 结果计算之前冻结"）。

- [ ] **Step 2：写 verifier 的失败测试**（`verify_b3_waiting_strict.py`）
  - 冻结 SHA 检查；
  - 角色检查：R/C/T 不相交，都是不同向量，无孪生；
  - 泄漏检查：把 T 的结果取反（`success[:, T, :] = ~success[:, T, :]`）后重新选择，choices 完全相同；
  - 合成回退：把 C 的成功表全部置为 False，certified 的所有选择都是 `fallback`，且 T 上用 120 min 打分；
  - 从零重算划分 0 与 7，与保存结果在 1e-12 内一致；
  - 由保存的逐划分数值独立重推 20 划分汇总与 E1-S1…S4。

Run: `python verify_b3_waiting_strict.py` → Expected: FAIL（结果文件不存在 / 模块缺失）

- [ ] **Step 3：实现 `b3_waiting_strict.py`**（`run_split` 复用 `estimator_probabilities(tables, success, noise, R, C/T, observed)`、`frontier.candidates`；`certified` 臂的候选按上面的顺序排列后交给 `strict_eval.certified_choice`；多进程用 `ProcessPoolExecutor`，`if __name__ == '__main__':` 保护，`OPENBLAS_NUM_THREADS=1`）。

- [ ] **Step 4：单次运行**

Run: `cd simulation/code && set OPENBLAS_NUM_THREADS=1 && python b3_waiting_strict.py`
Expected: 写出结果并打印各臂摘要。只运行一次；如果失败，记录失败原因，冻结修正版后再重跑。

- [ ] **Step 5：运行 verifier**

Run: `python verify_b3_waiting_strict.py` → Expected: `passed`

---

### Task 4: E2 每个校准接收机只提供一次探针

**Files:**
- Create: `simulation/configs/b3_waiting_budget_protocol.json` + freeze record
- Create: `simulation/code/b3_waiting_budget.py`, `simulation/code/verify_b3_waiting_budget.py`
- Output: `simulation/results/b3/waiting_budget_results.json`

**Interfaces:**
- Consumes: E1 的 `TEST_ROLES`、估计器概率、`strict_eval.cp_lower/binom_pvalue`
- Produces: `one_probe_labels(...)`、`isotonic_curve(...)`，Task 8 绘图用

**协议要点：**
- 角色与 E1 相同（R/C/T，划分 0–19，噪声 0.005）。预算 n ∈ {50, 100, 200}，每种历史从 C 中不放回抽取（两种历史各自独立抽取，同一向量可能同时出现在两种历史中，需写明）。
- 信息约束：每个校准接收机被随机分配到该规则族梯子上的一个候选 θ_i（均匀分配），只在 τ_i(θ_i) 探针一次，只得到 s_i = success(τ_i(θ_i))。估计器使用的是 R 的模型参考（模型仿真，不消耗探针预算）；阈值选择只用这 n 个一次性标签。用访问器 `LabelOracle` 实现：每个接收机只允许调用一次，再次调用即报错。
- 梯子（预先固定，各 20 个候选，按保守到激进排列）：固定等待 {120, 116, …, 44}；current、smoothed（τ=16，即全信息下 20/20 划分的选择，需写明）和 B3 belief 的阈值 {0.99, 0.98, …, 0.80}；B3 belief 加触发后延迟 δ（阈值固定为 0.90，δ ∈ {40, 38, …, 2}）。
- 估计与选择：
  - `plugin`：对 (梯子位置, s_i) 做保序回归（成功率随保守程度单调不减），选 Ŝ ≥ 0.9 的最激进候选；没有则在 120 min 探针。
  - `certified`：按梯子从保守到激进做固定序列检验。检验候选 θ_j 时，只用被分配到同样或更激进候选（位置 ≥ j）的接收机。依据单调性，这些接收机的成功概率都 ≤ S(θ_j)，所以 CP 下界保守有效。α = 0.025；不可行时回退到 120 min。
- 理想信息基准：同样 n 个接收机、每个接收机提供完整结果曲线时的同一选择规则（plugin 与 certified），以及 E1 中全部 316 个 C 的结果。
- 场景：基线（模型正确，主）；隐藏滞后 L=8 min（次；长历史的 plant 结果为模型在 τ−L 时的结果，梯子与估计器不变）。
- 指标（T 上）：逐历史成功率与 CP 下界、完成时间、相对同预算固定等待的配对差、认证成功的划分数。汇总 20 个划分，并标出预算不足、校准不可行和收益消失的区域。
- 预注册陈述：
  - E2-S1：每个预算下各规则 plugin 与 certified 的 T 上逐历史成功率（中位数）与满足 0.9 的划分数。
  - E2-S2：相对理想信息基准，完成时间收益保留的比例。
  - E2-S3：在隐藏滞后 8 min 下，有限探针重校准能否恢复逐历史 0.9。

- [ ] Step 1：写协议与冻结记录。
- [ ] Step 2：写 verifier 失败测试：`LabelOracle` 二次访问报错；`certified` 只使用位置 ≥ j 的接收机（构造一个合成例子，手算下界）；保序回归与 `sklearn` 无关的手写 PAVA 对拍；T 结果取反后选择不变；重算划分 0；汇总重推。运行，确认 FAIL。
- [ ] Step 3：实现 `b3_waiting_budget.py`（手写 PAVA；梯子位置 j 的 τ_i 用 `frontier.crossing_index`）。
- [ ] Step 4：单次运行。
- [ ] Step 5：verifier 通过。

---

### Task 5: E3 连接预测评价与等待决策

**Files:**
- Create: `simulation/configs/b3_predict_decide_protocol.json` + freeze record
- Create: `simulation/code/b3_predict_decide.py`, `simulation/code/verify_b3_predict_decide.py`
- Output: `simulation/results/b3/predict_decide_results.json`

**协议要点：**
- 角色同 E1（划分 0 为主，20 划分看稳定性；噪声 0.005）。信息集：到时刻 t 为止的有噪报告值序列和已知的指令历史。
- 目标量：恢复统计量 ρ_i(t) = rise_i(t)/naive_i，即式（2）在 W=20 窗口内的最大增量除以同一接收机的 naive 增量；成功等价于 ρ ≥ κ。
- 预测器（都在 R 上拟合，超参数在 R 内做 5 折选择）：
  1. `history_mean`：R 中同历史、同时刻 ρ 的均值，相当于固定等待所用的信息。
  2. `current`：对当前观测做核回归，带宽与 E1 相同。
  3. `smoothed`：对 EMA(τ=16) 做核回归。
  4. `arx`：岭回归，特征为 [y(t), y(t−2), …, y(t−8), t, 历史指示, 历史指示·t]，t<0 的滞后用 t=0 的值填充。
  5. `narx`：上述特征的二次多项式展开后做岭回归。
  6. `b3_belief`：B3 似然加权的 R 中 ρ 均值。
- 报告器预测：各预测器给出 10 min 后的无噪报告值 f(t+10)。history_mean 用同历史 R 的均值轨迹；current 和 smoothed 用持续值（y(t) 或 EMA）；arx 和 narx 另拟合一个 f(t+10) 回归；b3_belief 用加权轨迹。
- 转换：P(成功) = 1 − F̂_h(κ − ρ̂)，其中 F̂_h 是 R 内 5 折交叉拟合残差 ρ−ρ̂ 的经验分布（每种历史一个）。等待动作取运行最大概率首次 ≥ c 的时刻；c 在 C 上用 E1 的 certified 程序选择（梯子 0.99→0.50）。
- T 上的指标：(a) 10 min 报告器预测 RMSE（相当于"平均预测误差"）；(b) ρ 的 RMSE（全部决策时刻，以及 t∈[60,120] 的决策相关段）；(c) 成功概率的 Brier 分数和 10 箱可靠性误差 ECE；(d) 错误就绪率，即实际探针失败的比例，逐历史；(e) 带惩罚的完成时间、成功率及 CP 下界。
- 预注册读法：按 (a) 的排序与按 (e)（在都满足 0.9 的规则之间）的排序是否一致，如实报告一致或分离。实测数据只有实际执行探针的标签，因此 E3 不在实测数据上做反事实决策评价，并需写明这一点。

- [ ] Step 1：写协议与冻结记录。
- [ ] Step 2：verifier 失败测试：特征构造在 t 处不使用 t 之后的观测（因果性检查：把 t 之后的观测置为 NaN，预测不变）；残差分布只来自 R 的交叉拟合（把 C/T 的 ρ 置为 NaN，概率不变）；history_mean 预测与手算一致；从零重算划分 0。运行，确认 FAIL。
- [ ] Step 3：实现。
- [ ] Step 4：单次运行。
- [ ] Step 5：verifier 通过。

---

### Task 6: E4 任务定义与失配形式

**Files:**
- Create: `simulation/code/b3_window_tables.py`（输出 `simulation/results/b3/window_tables.npz`：naive_rise 与 rise，形状为 (3 个窗口, 2, 1000, 61)）
- Create: `simulation/configs/b3_waiting_scope_protocol.json` + freeze record
- Create: `simulation/code/b3_waiting_scope.py`, `simulation/code/verify_b3_waiting_scope.py`
- Output: `simulation/results/b3/waiting_scope_results.json`

**协议要点：**
- 探针仿真：时刻 [w, w+30]，1 min 网格，与 `b3_readiness.probe_rise` 同一求解器和容差；W=20 的上升量必须与 `readiness_tables.npz` 在 float32 精度内相同（verifier 检查）。运行前先计时 50 个作业，估算总耗时。
- 单因素敏感性（围绕 κ=0.5、W=20、D=120+W）：κ ∈ {0.4, 0.6, 0.8}；W ∈ {10, 30}；D ∈ {200, 260}。每个组合用 E1 的 certified 臂（划分 0 为主，20 个划分），报告 T 上逐历史成功率与下界、相对固定等待的配对完成时间差、oracle 可行性（T 上 oracle 两种历史都 ≥0.9）。κ=0.8 若不可行，照实保留。
- 失配形式（都只作用于 30 min 历史，作为压力测试；规则沿用 E1 在模型结果上 certified 选出的参数，即模型校准臂；另设"完整曲线、按 plant 结果认证"的重校准臂）：
  - 共同时间平移：L ∈ {4, 8, 16}（参照已有研究）。
  - 幅度下降：plant 的上升量为 a·rise，a ∈ {0.9, 0.8, 0.7}。
  - 恢复速度变慢：plant 的 ρ(t) = 模型的 ρ(t/s)，在 2 min 网格上线性插值，s ∈ {1.1, 1.25, 1.5}。
  - 接收机间差异：每个接收机有自己的速度因子 s_i = exp(σ z_i)，σ ∈ {0.1, 0.2}，z_i 用固定种子，作用于两种历史。
  - 以单个实测点锚定的变体（单独标注为"数据锚定"）：a* 使模型中位 ρ(60) 等于实测 0.0409；s* = 60/24.456；只报告 oracle 可行性和模型校准臂的结果。
- 读法：区分数据锚定的变化与人为压力测试；收益不稳健时缩小适用范围，不挑选情景。

- [ ] Step 1：实现 `b3_window_tables.py`，计时后运行（后台运行，监控进度；只结束自己启动的进程）。
- [ ] Step 2：写协议与冻结记录（写入 W 表的 SHA）。
- [ ] Step 3：verifier 失败测试：W=20 复现；s=1、a=1、L=0 时与基线完全相同；s 越大或 a 越小，模型校准臂的成功率单调不增（逐接收机检查）；重算划分 0。运行，确认 FAIL。
- [ ] Step 4：实现 `b3_waiting_scope.py`，单次运行。
- [ ] Step 5：verifier 通过。

---

### Task 7: 复现管线、参考指标与图表

**Files:**
- Modify: `scripts/reproduce.py`（full 步骤：`b3_window_tables`、`b3_waiting_strict`、`b3_waiting_budget`、`b3_predict_decide`、`b3_waiting_scope`；verify 步骤：两个审计、`verify_strict_eval` 和四个新 verifier；`revision8_metrics()` 展平新结果的 summary 与 statements；`compare_reference` 的通过列表加入新 verifier）
- Modify: `reproduction/reference_metrics.json`（加入新键；旧键不变）
- Modify: `scripts/build_paper_assets.py`：
  - 新表 II（`paper/tables/strict_waiting.tex`）：E1 划分 0、certified 臂；各规则逐历史成功率、95% 下界、完成时间、成功条件下完成时间，以及相对固定等待的差值与区间；脚注给出 20 个划分的稳定性。
  - 新图 2（`paper/figures/probe_budget.pdf/png`）：横轴为预算 n，纵轴分别为长历史成功率和平均完成时间；画 plugin 与 certified，以理想信息基准作为水平参考线；单栏宽。
  - 图 3(b)：把 "measured probe: ≥35.5 min" 改为 "≈35.5 min B3-equivalent shift (pure time shift)"，去掉不等号。
  - 补充材料表：协议追溯表、任务差异表、E3 指标表、E4 敏感性表。
- Modify: `simulation/code/pipeline_checks.py`（若需要：新表文件、新图）

- [ ] Step 1：写 `revision8_metrics()`；暂不更新参考文件，先运行 `python scripts/reproduce.py --mode verify`，确认因键不同而 FAIL（RED）。
- [ ] Step 2：用一个小脚本把新键加入 `reference_metrics.json`，并核对旧键完全不变；再次运行，确认 PASS。
- [ ] Step 3：生成图表，逐张查看渲染结果（标签重叠、溢出）。

---

### Task 8: 论文改写（≤7 页）与用语统一

**Files:** `paper/main.tex`, `paper/supplement.tex`, `paper/references.bib`（只在需要新增文献时追加，不改用户已有的条目）

- [ ] Step 1：贡献与摘要按方案第 1 节重写：(1) 实测数据中的信息价值，区分平均预测与下一指令窗口；(2) 明确仿真条件下的复用决策（E1 严格评价，报告不确定性，加上 E2 有限探针）；(3) 适用边界（失配与校准信息不足：图 3、E2 的不可行区域、E4）。不提前写入尚未完成的结果。
- [ ] Step 2：II 节加入抽样单位、信息集、κ 代理任务、失败处理；III 节加入角色分工、规则族、certified 校准与回退、预测到决策的转换（E3）；IV 节加入协议时间表简版、naive 分母、独立性边界；V 节按"预测信息价值 → 可靠等待（E1、E2）→ 预测与决策（E3）→ 失配与校准限制（图 3、E4）"组织；VI 节逐条对应已验证的贡献。
- [ ] Step 3：条件热图（旧图 2）和旧表 II（frontier 0.94）移到补充材料，作为"旧评价流程（含孪生样本）"对照，并引用审计的影响量。
- [ ] Step 4：用语检查。下面的命令应输出为空（"upper bound" 若仍用于其他含义，需逐条人工确认）：
  `grep -n -i "at least 35\|lower bound on the\|guarantee\|optimal stopping\|upper bound\|measured lower" paper/main.tex paper/supplement.tex`
- [ ] Step 5：编译主文与补充材料（0 warning）；主文 ≤ 7 页；运行 `pipeline_checks`（正文中 4 位以上小数都必须能在参考指标中找到）。
- [ ] Step 6：同步 README、Paper_Notes_ZH、Paper_Simulation_Map_ZH、Revision_Response_ZH（逐条回应方案中的 P0 与 E1–E4，以及方案第 6 节提交前检查表的每一项）。

---

### Task 9: 全量核验与独立审阅

- [ ] Step 1：`python scripts/reproduce.py --mode verify --compile-paper` 全部通过；刷新 `paper/qa_report.json`、`simulation/checksums.json`、`release_manifest.json`，确认所有哈希匹配、文件均为 LF。
- [ ] Step 2：派一个独立审阅子代理（全新上下文），对照方案第 6 节的检查表和本计划，审阅 main.tex、supplement.tex 与新结果 JSON，重点核查数字是否一致、有无超出证据的表述。逐条处理审阅意见。
- [ ] Step 3：用中文向用户汇报：更新了什么、主要结果（包括不利结果），以及仍未覆盖的内容（真实细胞上的反馈复用仍需实际执行所选策略后的独立实验）。询问是否 commit 和 push。

---

## 方案条目 → 任务对照（自查）

| 方案条目 | 任务 |
| --- | --- |
| P0 核实刺激协议 | Task 1 Step 1 |
| P0 35.5 min 解释 | Task 7（图 3）、Task 8 Step 4；协议核查结果一致，无需重算（Task 1 若发现不一致则重算） |
| P0 界定可靠性概率 | Global Constraints、Task 3 协议、Task 8 Step 2 |
| P0 分离方法选择与最终评价 | Task 1 Step 2（审计）、Task 3（R/C/T） |
| P0 处理不可行情况 | Task 1 Step 2、Task 2 回退测试、Task 3 |
| P0 统一实验与仿真任务 | Task 1 Step 2 任务差异表、Task 8 Step 2 |
| E1 | Task 2、Task 3 |
| E2 | Task 4 |
| E3 | Task 5（ARX、NARX 在内） |
| E4 | Task 6 |
| 六（七）页增删 | Task 7、Task 8 |
| 结论用语表 | Task 8 Step 4 |
| 提交前检查表 | Task 8 Step 6、Task 9 |
