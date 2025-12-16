# Agent Session Memory

This file captures standing collaboration preferences and process. Live task state belongs in `CURRENT_WORKSTREAM/README.md`.

## Working Docs

- Source of truth: `CURRENT_WORKSTREAM/README.md` (status, context, decisions)
- Checklist: `CURRENT_WORKSTREAM/TASKS.md`
- Artifacts: `CURRENT_WORKSTREAM/notes/`, `CURRENT_WORKSTREAM/scratch/`, `CURRENT_WORKSTREAM/exports/`
- Archive completed workstreams to `docs/workstreams/<utc-timestamp>/` and then recreate a fresh `CURRENT_WORKSTREAM/` skeleton

## Codebase Index

- Package root: `src/palimpzest/`
   - Public API surface: `src/palimpzest/__init__.py` (re-exports the primary user-facing classes)
   - Core data + schemas: `src/palimpzest/core/` (datasets/contexts/schemas, execution + optimizer plumbing)
   - Query pipeline: `src/palimpzest/query/` (operators/generators/execution/optimizer/processor)
   - Policies: `src/palimpzest/policy.py` (cost/quality/time policy definitions)
   - Agents: `src/palimpzest/agents/` (compute/search agents)
   - Prompts: `src/palimpzest/prompts/` (prompt templates + prompt factory)
   - Tools: `src/palimpzest/tools/` (PDF and other tooling integrations)
   - Validation: `src/palimpzest/validator/` (validation entry points)
   - Utilities: `src/palimpzest/utils/` (env/hash/model helpers, progress, UDF utilities)

- Examples: `demos/` (end-to-end scripts showing typical usage)
- Tests: `tests/pytest/` (pytest suite; `conftest.py` + focused test modules)
- Docs site: `website/` (Docusaurus docs + blog)
- Research scripts: `abacus-research/` (workload-specific experiments/ablation scripts)
- Test fixtures/data: `testdata/` (download scripts + sample datasets)

## Working Style

- Keep UX/copy direct; no hidden behavior.
- Prefer fail-fast code and explicit errors over silent fallbacks.
- Ask 1–3 clarifying questions when requirements are ambiguous.
- Run fast, relevant tests (e.g. `pytest …`) before calling work complete.
- Avoid low-signal tests; add tests only when they meaningfully reduce regressions.

## Tooling Expectations

- No bespoke, ad-hoc bash for investigations.
- For investigation work, write a small, reusable script under `scratchpad/scripts/` (document inputs/outputs).
- Apply all file edits via the Codex `apply_patch` tool.

## Parallel Agents (VS Code + git worktrees)

This repo supports running multiple parallel agents via VS Code agent sessions. The workflow below is optimized for:

- minimal env cross-talk (each worktree has its own venv)
- predictable integration (merge commits; no rebases in shared branches)
- small, frequent syncs to reduce conflict size

### One agent session per worktree

- Create one git worktree per agent.
- Open each worktree in its own VS Code window (recommended). Avoid multi-root workspaces for agent sessions unless you have a strong reason—Source Control and interpreter selection can become ambiguous.
- Run exactly one agent session per VS Code window/worktree.

Example worktree layout (sibling folders):

```bash
git worktree add ../palimpzest.wt-agent-a -b agent/a
git worktree add ../palimpzest.wt-agent-b -b agent/b
git worktree add ../palimpzest.wt-integrate -b integrate/active
```

### Per-worktree Python environment (venv)

Each worktree maintains its own `.venv/` at the worktree root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

In the corresponding VS Code window:

- Run **Python: Select Interpreter** and choose `${workspaceFolder}/.venv/bin/python`.
- To make interpreter selection sticky and reduce mistakes, prefer committing a per-worktree setting:

Create `.vscode/settings.json` in that worktree:

```json
{
   "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
   "python.terminal.activateEnvironment": true
}
```

Notes:

- Do not share a single venv across worktrees; it causes non-deterministic failures when dependencies diverge.
- `pip` caching should keep multi-venv installs reasonably fast.

### Integration policy: merge commits

This repo prefers merge commits for shared branches. The goal is to avoid rewriting history that other agents are building on.

Agent sync loop (do this frequently, keeping diffs small):

```bash
git fetch origin
git merge origin/main
```

Recommended global config to reduce repeat-conflict pain:

```bash
git config --global rerere.enabled true
```

### Optional: dedicated integration worktree

To reduce "everyone merges at the end" conflicts:

- Create a dedicated integration worktree/branch (e.g. `integrate/active`).
- Only the integration agent merges other agent branches into it.
- The integration agent runs the merge gate (tests/lint) and resolves conflicts once, centrally.

Suggested merge cadence:

- Each agent merges `origin/main` into their branch frequently.
- The integration agent merges agent branches into `integrate/active` frequently.
- `main` stays green; land changes via PR (or direct merge if you’re solo), but preserve merge commits.

### Merge hygiene

- Prefer task/area ownership: avoid having multiple agents edit the same files at the same time.
- If overlap is unavoidable, agree on the interface/contract first (function signatures, schema, output shape).
- Keep PRs/branches small; conflicts scale superlinearly with diff size.

## Local Environment

- Python: 3.12+ (see `pyproject.toml`).
- Secrets: use Doppler (preferred).
   - Authenticate once: `doppler login`
   - Configure repo (if needed): `doppler setup` (choose project + config)
   - Run with injected env: `doppler run -- <command>`
- Expected env vars depend on provider; common ones in this repo:
   - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, `VLLM_API_KEY`
- Notes:
   - Some scripts also load a local `.env` via `python-dotenv` / `palimpzest.utils.env_helpers`; treat `.env` as an escape hatch, never committed.

## Quick Commands

- Install (editable): `pip install -e .`
- Tests: `pytest -q`
- Lint: `ruff check .`
- Format (if/when used): `ruff format .`
- Run a demo (with secrets): `doppler run -- python demos/simple-demo.py`

## Definition of Done

- Repro/goal captured in `CURRENT_WORKSTREAM/README.md` (incl. exact command).
- Change is minimal, fail-fast, and has clear errors.
- Tests: updated/added only when they reduce regression risk; relevant `pytest` is green.
- Quality: `ruff` passes on touched code.
- If user-facing behavior changes: update the closest demo/docs snippet.

## Workstream Lifecycle

1. Spin up
   - Ensure `CURRENT_WORKSTREAM/` exists with `README.md` + `TASKS.md`.
   - Record goal, success criteria, and immediate hypotheses in `README.md`.

2. Execute / Debug
   - Reproduce first; capture command + environment in `README.md`.
   - Save logs/traces under `exports/` and keep findings summarized in `README.md`.
   - Prefer repeatable steps and scripts over manual poking.

3. Wrap up
   - Summarize root cause, fix, and verification evidence in `README.md`.
   - Mark `TASKS.md` complete (leave explicit follow-ups if any).
   - Archive the entire workstream directory to `docs/workstreams/<utc-timestamp>/`.
