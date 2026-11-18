
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

SHAPES: tuple[int, ...] = (512, 1024, 2048, 4096)
CPU_SHAPES: tuple[int, ...] = (256, 512, 768, 1024)
REPEATS = 10
PROFILES = {"gpu": SHAPES, "cpu": CPU_SHAPES}
DEFAULT_REPEATS_BY_PROFILE = {"gpu": REPEATS, "cpu": REPEATS}

E1_VARIANTS: tuple[str, ...] = (
    "full",
    "minus_state_in_prompt",
    "intent_only_control",
    "minus_local_checks",
    "minus_stage_ordering",
)

STAGE_VARIANTS = ("minus_feedback_optimization", "minus_parameter_search")

MODEL_ROLES: tuple[str, ...] = (
    "qwen3.5-397b-a17b",
    "glm-5.2",
    "deepseek-v4.1-flash",
)
MODEL_REUSES_E1_FULL = {"qwen3.5-397b-a17b"}

E3_ARMS = ("full", "iterative_compile_validate_feedback")
E3_SHAPES: tuple[int, ...] = (512, 1024)

D1_ARMS = ("adaptive", "frozen")
D1_REPEATS = 10

REHEARSAL_SHAPE = 64
REHEARSAL_REPEATS = 2

BLOCK_NAMES = {
    "E1": "components",
    "E2": "models",
    "E3": "budget",
    "D1": "dependency",
    "PILOT": "pilot",
    "REHEARSAL": "rehearsal",
}


@dataclass
class Task:
    task_id: str
    block: str
    kind: str
    shape: tuple[int, int, int]
    repeat: int
    variant: str | None = None
    model_role: str | None = None
    arm: str | None = None
    parent_task_id: str | None = None
    reuse_of: str | None = None
    coefficients_only: bool = False
    notes: str = ""
    order_index: int = -1
    planned_status: str = "planned"

    def as_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "block": self.block,
            "kind": self.kind,
            "shape_M": self.shape[0],
            "shape_N": self.shape[1],
            "shape_K": self.shape[2],
            "shape": list(self.shape),
            "repeat": self.repeat,
            "variant": self.variant,
            "model_role": self.model_role,
            "arm": self.arm,
            "parent_task_id": self.parent_task_id,
            "reuse_of": self.reuse_of,
            "coefficients_only": self.coefficients_only,
            "notes": self.notes,
            "order_index": self.order_index,
            "planned_status": self.planned_status,
        }


def _shape_tuple(shape: int) -> tuple[int, int, int]:
    return (int(shape), int(shape), int(shape))


def e1_tasks(shapes: Sequence[int] = SHAPES, repeats: int = REPEATS) -> list[Task]:
    tasks: list[Task] = []
    for variant in E1_VARIANTS:
        for shape in shapes:
            for repeat in range(1, int(repeats) + 1):
                tasks.append(Task(
                    task_id=f"E1-{variant}-{shape}-r{repeat:02d}",
                    block="E1", kind="independent", shape=_shape_tuple(shape),
                    repeat=repeat, variant=variant,
                    notes=("reference configuration for the component ablation"
                           if variant == "full" else
                           "declared ablation; only the declared factor may differ"),
                ))
    return tasks


def e1_full_task_id(shape: int, repeat: int) -> str:
    return f"E1-full-{int(shape)}-r{int(repeat):02d}"


def e2_tasks(shapes: Sequence[int] = SHAPES, repeats: int = REPEATS,
             model_roles: Sequence[str] = MODEL_ROLES) -> list[Task]:
    tasks: list[Task] = []
    for role in model_roles:
        for shape in shapes:
            for repeat in range(1, int(repeats) + 1):
                reuse = role in MODEL_REUSES_E1_FULL
                parent = e1_full_task_id(shape, repeat) if reuse else None
                tasks.append(Task(
                    task_id=(parent if reuse else
                             f"E2-{_role_slug(role)}-{shape}-r{repeat:02d}"),
                    block="E2",
                    kind="alias" if reuse else "independent",
                    shape=_shape_tuple(shape), repeat=repeat,
                    variant="full", model_role=role,
                    reuse_of=parent,
                    notes=("pre-assigned reuse of the E1 Full slot; no new N and "
                           "no new cost" if reuse else
                           "independent generation for this model"),
                ))
    return tasks


def _role_slug(role: str) -> str:
    return role.replace(".", "-").replace(" ", "-")


FEEDBACK_STAGE_CPU_NOTE = (
    "P2 (feedback optimization) is disabled by the frozen method when the target "
    "backend is cpu (app.py:auto_unlock_performance), so a without_feedback branch "
    "can never differ from its parent on CPU and is not planned")


def stage_branch_tasks(shapes: Sequence[int] = SHAPES,
                       repeats: int = REPEATS,
                       backend: str = "cuda") -> list[Task]:
    tasks: list[Task] = []
    branch_applicable = str(backend) != "cpu"
    for shape in shapes:
        for repeat in range(1, int(repeats) + 1):
            parent = e1_full_task_id(shape, repeat)
            tasks.append(Task(
                task_id=f"E1-woP3-{shape}-r{repeat:02d}",
                block="E1", kind="alias", shape=_shape_tuple(shape), repeat=repeat,
                variant="minus_parameter_search", parent_task_id=parent,
                notes="P2 checkpoint reused as the seed; only P3 runs, timed "
                      "independently; not a new generation",
            ))
            if branch_applicable:
                tasks.append(Task(
                    task_id=f"E1-woFB-{shape}-r{repeat:02d}",
                    block="E1", kind="branch", shape=_shape_tuple(shape),
                    repeat=repeat,
                    variant="minus_feedback_optimization", parent_task_id=parent,
                    notes="branch derived from the parent's P1 selected checkpoint; "
                          "skips P2 and runs the same P3 protocol; counts as a paired "
                          "observation, not a new N",
                ))
    return tasks


def e3_tasks(shapes: Sequence[int] = E3_SHAPES,
             repeats: int = REPEATS) -> list[Task]:
    tasks: list[Task] = []
    for arm in E3_ARMS:
        for shape in shapes:
            for repeat in range(1, int(repeats) + 1):
                tasks.append(Task(
                    task_id=f"E3-{arm}-{shape}-r{repeat:02d}",
                    block="E3", kind="independent", shape=_shape_tuple(shape),
                    repeat=repeat, arm=arm,
                    notes="strict equal wall-clock budget from the first request",
                ))
    return tasks


def d1_tasks(shapes: Sequence[int] = SHAPES,
             repeats: int = D1_REPEATS) -> list[Task]:
    tasks: list[Task] = []
    for arm in D1_ARMS:
        for shape in shapes:
            for repeat in range(1, int(repeats) + 1):
                tasks.append(Task(
                    task_id=f"D1-{arm}-{shape}-r{repeat:02d}",
                    block="D1", kind="independent", shape=_shape_tuple(shape),
                    repeat=repeat, arm=arm,
                    notes="conditional extension: needs scope-exp-dependency-v2",
                ))
    return tasks


def pilot_tasks(shapes: Sequence[int] = SHAPES,
                model_roles: Sequence[str] = MODEL_ROLES) -> list[Task]:
    tasks: list[Task] = []
    order = 1
    for shape in shapes:
        for repeat in (1, 2):
            tasks.append(Task(
                task_id=f"PILOT-full-{shape}-r{repeat:02d}", block="PILOT",
                kind="independent", shape=_shape_tuple(shape), repeat=repeat,
                variant="full", model_role=MODEL_ROLES[0],
                order_index=order,
                notes="pilot data stays in the development split; excluded from "
                      "formal statistics"))
            order += 1
    for role in list(model_roles)[1:]:
        first_shape = int(shapes[0]) if shapes else int(SHAPES[0])
        tasks.append(Task(
            task_id=f"PILOT-{_role_slug(role)}-{first_shape}-r01", block="PILOT",
            kind="independent", shape=_shape_tuple(first_shape), repeat=1,
            variant="full", model_role=role, order_index=order,
            notes="pilot: provider behaviour and cost for this model"))
        order += 1
    return tasks


def shapes_for_blocks(blocks: Iterable[str]) -> list[int]:
    shapes: set[int] = set()
    wanted = {block.upper() for block in blocks}
    if wanted & {"E1", "E2", "D1", "PILOT"}:
        shapes.update(SHAPES)
    if "E3" in wanted:
        shapes.update(E3_SHAPES)
    if "REHEARSAL" in wanted:
        shapes.add(REHEARSAL_SHAPE)
    return sorted(shapes)


def rehearsal_tasks() -> list[Task]:
    tasks: list[Task] = []
    for repeat in range(1, REHEARSAL_REPEATS + 1):
        tasks.append(Task(
            task_id=f"REH-{REHEARSAL_SHAPE}-r{repeat:02d}", block="REHEARSAL",
            kind="independent", shape=_shape_tuple(REHEARSAL_SHAPE), repeat=repeat,
            variant="full",
            notes="pipeline rehearsal: the repository skeleton is evaluated and timed "
                  "to prove the harness works; never a SCOPE generation result"))
    return tasks


def block_selection(blocks: Iterable[str], *, shapes: Sequence[int] | None = None,
                    repeats: int = REPEATS,
                    model_roles: Sequence[str] = MODEL_ROLES,
                    backend: str = "cuda") -> list[Task]:
    wanted = {block.upper() for block in blocks}
    shapes = list(shapes) if shapes else None
    core_shapes = shapes if shapes is not None else list(SHAPES)
    budget_shapes = shapes if shapes is not None else list(E3_SHAPES)
    tasks: list[Task] = []
    if "E1" in wanted:
        tasks.extend(e1_tasks(core_shapes, repeats))
        tasks.extend(stage_branch_tasks(core_shapes, repeats, backend))
    if "E2" in wanted:
        tasks.extend(e2_tasks(core_shapes, repeats, model_roles))
    if "E3" in wanted:
        tasks.extend(e3_tasks(budget_shapes, repeats))
    if "D1" in wanted:
        tasks.extend(d1_tasks(core_shapes, repeats))
    if "PILOT" in wanted:
        tasks.extend(pilot_tasks(core_shapes, model_roles))
    if "REHEARSAL" in wanted:
        tasks.extend(rehearsal_tasks())
    return tasks


def interleave(tasks: Sequence[Task]) -> list[Task]:
    ordered = list(tasks)
    groups: dict[str, list[Task]] = {}
    for task in ordered:
        key = "|".join(str(part) for part in
                       (task.block, task.variant, task.model_role, task.arm))
        groups.setdefault(key, []).append(task)
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda task: (task.shape, task.repeat,
                                                           task.task_id))
    queues = [groups[key] for key in sorted(groups)]
    cursor = 0
    while any(queues):
        for queue in queues:
            if queue:
                queue.pop(0).order_index = cursor
                cursor += 1
        queues = [queue for queue in queues if queue]
    return sorted(ordered, key=lambda task: task.order_index)


def build_task_matrix(*, blocks: Sequence[str] = ("E1", "E2"),
                      shapes: Sequence[int] | None = None,
                      repeats: int | None = None,
                      model_roles: Sequence[str] | None = None,
                      backend: str = "cuda",
                      include_branches: bool = True,
                      include_aliases: bool = True) -> list[Task]:
    shapes = list(shapes) if shapes else None
    repeats = int(repeats) if repeats else REPEATS
    model_roles = list(model_roles) if model_roles else MODEL_ROLES
    tasks = block_selection(blocks, shapes=shapes, repeats=repeats,
                            model_roles=model_roles, backend=backend)
    if not include_branches:
        tasks = [task for task in tasks if not (task.block == "E1" and task.kind == "branch")]
    if not include_aliases:
        tasks = [task for task in tasks if not (task.block == "E1" and task.kind == "alias")]
    return interleave(tasks)


def _count(tasks: Iterable[Task], **match: Any) -> int:
    total = 0
    for task in tasks:
        if all(getattr(task, key) == value for key, value in match.items()):
            total += 1
    return total


def expected_counts(blocks: Sequence[str] = ("E1", "E2"),
                    shapes: Sequence[int] | None = None,
                    repeats: int | None = None,
                    model_roles: Sequence[str] | None = None,
                    backend: str = "cuda") -> dict[str, int]:
    wanted = {block.upper() for block in blocks}
    explicit = list(shapes) if shapes else None
    core_shapes = explicit if explicit is not None else list(SHAPES)
    budget_shapes = explicit if explicit is not None else list(E3_SHAPES)
    repeats = int(repeats) if repeats else REPEATS
    model_roles = list(model_roles) if model_roles else MODEL_ROLES
    reused = [role for role in model_roles if role in MODEL_REUSES_E1_FULL]
    counts: dict[str, int] = {}
    if "E1" in wanted:
        counts["E1_independent"] = len(E1_VARIANTS) * len(core_shapes) * repeats
        counts["E1_branches_no_feedback"] = (len(core_shapes) * repeats
                                             if str(backend) != "cpu" else 0)
        counts["E1_aliases_no_parameter_search"] = len(core_shapes) * repeats
    if "E2" in wanted:
        counts["E2_independent_new"] = ((len(model_roles) - len(reused))
                                        * len(core_shapes) * repeats)
        counts["E2_aliases_reused"] = len(reused) * len(core_shapes) * repeats
    if "E3" in wanted:
        counts["E3_independent"] = len(E3_ARMS) * len(budget_shapes) * repeats
    if "D1" in wanted:
        counts["D1_independent"] = len(D1_ARMS) * len(core_shapes) * repeats
    if "PILOT" in wanted:
        counts["PILOT_independent"] = (len(core_shapes) * 2
                                       + max(0, len(model_roles) - 1))
    if "REHEARSAL" in wanted:
        counts["REHEARSAL_independent"] = REHEARSAL_REPEATS
    if "E1" in wanted and "E2" in wanted:
        counts["core_independent_total"] = (counts["E1_independent"]
                                            + counts["E2_independent_new"])
    return counts


def validate_matrix(tasks: Sequence[Task], blocks: Sequence[str] = ("E1", "E2"),
                    *, shapes: Sequence[int] | None = None,
                    repeats: int | None = None,
                    model_roles: Sequence[str] | None = None,
                    backend: str = "cuda") -> dict[str, Any]:
    expected = expected_counts(blocks, shapes=shapes, repeats=repeats,
                              model_roles=model_roles, backend=backend)
    allowed_shapes = set(int(shape) for shape in
                         (list(shapes) if shapes
                          else list(SHAPES) + list(E3_SHAPES)))
    observed: dict[str, int] = {}
    if "E1" in {b.upper() for b in blocks}:
        observed["E1_independent"] = _count(tasks, block="E1", kind="independent")
        observed["E1_branches_no_feedback"] = _count(tasks, block="E1", kind="branch")
        observed["E1_aliases_no_parameter_search"] = _count(tasks, block="E1", kind="alias")
    if "E2" in {b.upper() for b in blocks}:
        observed["E2_independent_new"] = _count(tasks, block="E2", kind="independent")
        observed["E2_aliases_reused"] = _count(tasks, block="E2", kind="alias")
    if "E3" in {b.upper() for b in blocks}:
        observed["E3_independent"] = _count(tasks, block="E3", kind="independent")
    if "D1" in {b.upper() for b in blocks}:
        observed["D1_independent"] = _count(tasks, block="D1", kind="independent")
    if "PILOT" in {b.upper() for b in blocks}:
        observed["PILOT_independent"] = _count(tasks, block="PILOT", kind="independent")
    if "REHEARSAL" in {b.upper() for b in blocks}:
        observed["REHEARSAL_independent"] = _count(tasks, block="REHEARSAL",
                                                   kind="independent")
    if "E1" in {b.upper() for b in blocks} and "E2" in {b.upper() for b in blocks}:
        observed["core_independent_total"] = (observed["E1_independent"]
                                              + observed["E2_independent_new"])

    problems: list[str] = []
    for key, value in expected.items():
        if observed.get(key) != value:
            problems.append(f"{key}: expected {value}, observed {observed.get(key)}")
    ids = [task.task_id for task in tasks]
    by_id: dict[str, list[Task]] = {}
    for task in tasks:
        by_id.setdefault(task.task_id, []).append(task)
    duplicates: list[str] = []
    for task_id, group in by_id.items():
        if len(group) <= 1:
            continue
        real = [task for task in group if task.kind != "alias" or task.reuse_of != task_id]
        aliases = [task for task in group if task not in real]
        if len(real) != 1 or len(aliases) != len(group) - 1:
            duplicates.append(task_id)
    if duplicates:
        problems.append(f"duplicate task ids without an explicit reuse relation: "
                        f"{duplicates[:8]}")
    for task in tasks:
        M, N, K = task.shape
        if not (M > 0 and N > 0 and K > 0):
            problems.append(f"{task.task_id}: bad shape {task.shape}")
        if allowed_shapes and M not in allowed_shapes:
            problems.append(f"{task.task_id}: shape {M} is not in the frozen "
                            f"shape list {sorted(allowed_shapes)}")
        if task.kind == "branch" and not task.parent_task_id:
            problems.append(f"{task.task_id}: branch without a parent")
        if task.kind == "alias" and task.block == "E1" and not task.parent_task_id:
            problems.append(f"{task.task_id}: alias without a parent")
    if problems:
        raise AssertionError("task matrix validation failed: " + "; ".join(problems))
    return {"expected": expected, "observed": observed, "problems": problems,
            "task_count": len(tasks), "blocks": list(blocks)}
