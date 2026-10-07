"""Headless eval runner: tasks x configs x seeds -> success table judged on ground truth.

Built to be cheap to experiment with:
  * `--dry-run` prints the estimated cost and exits; paid models also require `--yes`
  * `--budget-usd` caps total spend, `--run-budget` caps any single run
  * results append to a JSONL file; re-running skips finished (model, effort, config, task, seed)
  * default is 1 config x 3 seeds; the scripted baseline is free and needs no key

    python -m scout.warehouse.evals                                   # free baseline
    python -m scout.warehouse.evals --model claude-haiku-4-5 --dry-run
    python -m scout.warehouse.evals --model claude-sonnet-5-5 --effort low --seeds 2 --yes
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from statistics import mean

from scout.harness.trace import Trace
from scout.warehouse.agent import run_task
from scout.warehouse.embodiment import Embodiment
from scout.warehouse.evaluator import Evaluator
from scout.warehouse.models import Usage, get_spec
from scout.warehouse.planners import BudgetExceeded, make_planner
from scout.warehouse.tasks import TASKS, Task
from scout.warehouse.world import Warehouse

CONFIGS = {
    "full":             dict(use_evaluator=True, false_success=0.0),
    "judge_fs=20%":     dict(use_evaluator=True, false_success=0.2),
    "judge_fs=50%":     dict(use_evaluator=True, false_success=0.5),
    "no_evaluator":     dict(use_evaluator=False, false_success=0.0),
}

# Rough per-run cost model (see the estimate in the chat): ~1k-token fixed prefix, ~220 tokens added
# per turn, prompt caching assumed to work. The real number comes from usage logged per run.
EST_TURNS, EST_ADD, EST_FIXED = 18, 220, 1100


def estimate_run_cost(model_id: str, out_per_turn: int = 600) -> float | None:
    spec = get_spec(model_id)
    if spec.provider == "baseline":
        return 0.0
    u, ctx = Usage(), EST_FIXED
    for t in range(EST_TURNS):
        u.cache_read += ctx if t else 0
        u.cache_write += EST_ADD if t else ctx
        ctx += EST_ADD + out_per_turn
        u.output += out_per_turn
    return u.cost_usd(spec)


def _key(r: dict) -> tuple:
    return (r["model"], r.get("effort"), r["config"], r["task"], r["seed"])


async def run_one(task: Task, cname: str, cfg: dict, seed: int, model: str, effort: str | None,
                  run_budget: float | None, trace_dir: str) -> dict:
    w = Warehouse(seed=seed)
    if task.setup:
        task.setup(w)
    emb = Embodiment(w, Evaluator(false_success_rate=cfg["false_success"], seed=seed),
                     use_evaluator=cfg["use_evaluator"],
                     user_reply=lambda q: "Please do not move heavy items; skip it.")
    planner = make_planner(model, goals=task.goals, effort=effort, budget_usd=run_budget)

    async def emit(_):
        pass

    base = {"model": model, "effort": effort, "config": cname, "task": task.id, "seed": seed}
    try:
        summ = await run_task(task.instruction, emb, planner, Trace(trace_dir, name=f"{model}-{cname}-{task.id}-s{seed}"),
                              emit, max_turns=task.max_turns)
        ok = task.success(w) and (summ["asked_user"] if task.expects_user_query else True)
        err = None
    except BudgetExceeded as e:
        summ, ok, err = {"turns": 0}, False, f"aborted: {e}"
    usage = getattr(planner, "usage", Usage())
    cost = usage.cost_usd(get_spec(model))
    return base | {"ok": ok, "turns": summ["turns"], "sim_time": round(w.t, 1), "error": err,
                   "usage": usage.__dict__, "cost_usd": cost, **emb.stats}


def print_table(rows: list[dict], configs: list[str], task_ids: list[str]) -> None:
    print(f"\n{'config':<14}" + "".join(f"{i[:13]:>15}" for i in task_ids) + f"{'overall':>10}{'turns':>8}")
    for c in configs:
        cells, allr = [], []
        for i in task_ids:
            r = [x for x in rows if x["config"] == c and x["task"] == i]
            allr += r
            cells.append(f"{100 * mean(x['ok'] for x in r):>14.0f}%" if r else f"{'-':>15}")
        if allr:
            print(f"{c:<14}" + "".join(cells) + f"{100 * mean(x['ok'] for x in allr):>9.0f}%{mean(x['turns'] for x in allr):>8.1f}")


async def main_async(args) -> int:
    spec = get_spec(args.model)
    tasks = [t for t in TASKS if (not args.tasks or t.id in args.tasks)]
    if spec.provider == "baseline":
        tasks = [t for t in tasks if t.goals]  # the scripted baseline has no NL understanding
    cfgs = {k: v for k, v in CONFIGS.items() if k in args.configs}
    results_path = Path(args.results or f"runs/evals/{args.model}.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = [json.loads(l) for l in results_path.read_text().splitlines()] if results_path.exists() else []
    done_keys = {_key(r) for r in done}
    todo = [(c, t, s) for c in cfgs for t in tasks for s in range(args.seeds)
            if (args.model, args.effort, c, t.id, s) not in done_keys]

    per_run = estimate_run_cost(args.model)
    est = None if per_run is None else per_run * len(todo)
    print(f"model={args.model} effort={args.effort or 'default'} provider={spec.provider}"
          + ("" if spec.verified else "  [UNVERIFIED id/price]"))
    print(f"{len(todo)} runs to do ({len(done_keys)} already done and skipped)")
    if spec.provider != "baseline":
        if est is None:
            print("estimated cost: unknown (no price in registry; add one in models.local.json)")
        else:
            print(f"estimated cost: ~${est:.2f}  (~${per_run:.2f}/run, rough; caps: total ${args.budget_usd:.2f}, per-run ${args.run_budget:.2f})")
        if not spec.available():
            print(f"missing API key for provider '{spec.provider}'; set it in the environment first")
            return 2
    if args.dry_run:
        return 0
    if spec.provider != "baseline" and not args.yes:
        print("paid model: re-run with --yes to spend real money")
        return 1

    trace_dir = tempfile.mkdtemp(prefix="evals-")
    spent, new = 0.0, []
    for c, t, s in todo:
        if spec.provider != "baseline" and spent >= args.budget_usd:
            print(f"total budget ${args.budget_usd:.2f} reached; stopping (rerun to resume)")
            break
        try:
            r = await run_one(t, c, cfgs[c], s, args.model, args.effort, args.run_budget, trace_dir)
        except Exception as e:  # API/auth errors: fail loudly on the first run, keep going afterwards
            if not new:
                raise
            print(f"  run {c}/{t.id}/s{s} failed: {type(e).__name__}: {e}")
            continue
        spent += r["cost_usd"] or 0.0
        new.append(r)
        with results_path.open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"  {c:<13}{t.id:<20}s{s}  {'ok ' if r['ok'] else 'FAIL'}  turns={r['turns']:<3}"
              + (f" cost=${r['cost_usd']:.3f}" if r["cost_usd"] is not None else "") + (f"  {r['error']}" if r["error"] else ""))

    rows = [r for r in done + new if r["model"] == args.model and r.get("effort") == args.effort]
    print_table(rows, list(cfgs), [t.id for t in tasks])
    print(f"\nspent this session: ${spent:.3f}   results: {results_path}   traces: {trace_dir}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="baseline", help="registry id (see scout/warehouse/models.py) or any model id")
    p.add_argument("--effort", default=None)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--configs", nargs="*", default=["full"], choices=list(CONFIGS))
    p.add_argument("--budget-usd", type=float, default=3.0, help="stop after this much total spend")
    p.add_argument("--run-budget", type=float, default=0.75, help="abort any single run past this spend")
    p.add_argument("--results", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true", help="confirm spending real money")
    raise SystemExit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
