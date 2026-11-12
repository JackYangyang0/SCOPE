# scope_exp

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate              # Linux
# .venv\Scripts\activate               # Windows
pip install -r requirements.txt
```

依赖只有 `numpy` 与 `pyyaml`。按后端另有要求：

| 后端 | 额外要求 |
|---|---|
| CPU | `gcc`（Linux）或 `cl` / `vcvars64`（Windows） |
| CUDA | NVIDIA 驱动 + CUDA Toolkit（`nvcc`）；门禁可选 `compute-sanitizer` |
| 生成 | `conf.yaml` 中的 provider key（见下） |



### 正式流程（需要配置conf.yaml）

```bash
python -m scope_exp prepare  --campaign scope-core --blocks E1 E2 --backend cuda
python -m scope_exp selfcheck --campaign scope-core
python -m scope_exp controls --campaign scope-core --backend cuda
python -m scope_exp ablation-audit --campaign scope-core
python -m scope_exp freeze   --campaign scope-core --per-task-wall-seconds 28800
python -m scope_exp pilot    --campaign scope-core
python -m scope_exp run      --campaign scope-core
python -m scope_exp summarize --campaign scope-core --figures
python -m scope_exp audit    --campaign scope-core --deep
```

###  CPU only

```bash
python -m scope_exp prepare  --campaign scope-cpu --blocks E1 PILOT --backend cpu \
                             --profile cpu --dataset-jobs 4
python -m scope_exp controls --campaign scope-cpu --backend cpu
python -m scope_exp freeze   --campaign scope-cpu --blocks E1 PILOT --backend cpu \
                             --shapes 256 512 768 1024 --repeats 10 \
                             --models <conf.yaml 中的model值> \
                             --per-task-wall-seconds 28800
python -m scope_exp pilot    --campaign scope-pilot --task-ids PILOT-full-256-r01
python -m scope_exp run      --campaign scope-cpu --blocks E1
python -m scope_exp summarize --campaign scope-cpu --figures
python -m scope_exp audit    --campaign scope-cpu
```

## 产物

```
results/<campaign>/
├── FROZEN_PROTOCOL.json       冻结的协议（任务矩阵、判据、预算、快照哈希）
├── datasets/*/manifest.json   输入清单（种子 + 逐文件 sha256）
├── audit/controls_gate.json   门禁记录（含 sanitizer 工具路径与结论）
├── analysis/
│   ├── completeness.csv       每个计划格的状态
│   ├── correctness.csv        每个任务的主检查/系数/判定
│   ├── rq2_components.csv     变体 × 形状聚合
│   ├── stage_pairs.csv        阶段配对
│   └── RESULTS_zh.md          人读报告
└── tasks/<task_id>/
    ├── status.json            状态、交付阶段、验证摘要、成本
    ├── validation.csv         逐用例误差（64 + 128 行）
    ├── checkpoints/<phase>/<hash>/kernel/   交付内核源码 + manifest
    └── app_evidence/          方法侧证据（终态链、诊断、最终代码）
```

---

results中`_workspace_`文件夹是运行时产生的临时目录
