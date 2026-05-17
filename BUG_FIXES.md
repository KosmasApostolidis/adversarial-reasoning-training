# Bug Fixes — Phase 3 Report

Phase 2 of the bug-hunt mandate applied all 11 approved fixes (B01–B11). Each fix is a single commit with TDD-driven test coverage and a behaviour-preservation contract documented in the commit body.

## Aggregate

| Metric | Value |
|---|---|
| Findings opened in `BUG_HUNT.md` | 11 |
| Findings approved for Phase 2 | 11 (B01–B11) |
| Fixes shipped | 11 |
| Commits | 11 (one per fix) |
| New tests added | 20 |
| Existing tests modified | 1 (`test_seed.py::test_seed_everything_sets_pythonhashseed` replaced by `test_seed_everything_does_not_touch_pythonhashseed` — the prior test pinned down B07's buggy no-op behaviour) |
| Existing tests weakened | 0 |
| Pre-existing test failures retained as-is | 2 (`test_ablation_schemas.py`, unrelated to this hunt — see "Out of scope" below) |
| Full suite excluding those 2 failures | 370 passed |

By class: correctness 4, numerical-reproducibility 2, API-contract 3, domain/statistical 1, resource/security-adjacent 1.

By severity: 7 HIGH-confidence, 3 HIGH-impact-MED-confidence, 1 LOW-impact / latent.

---

## Per-fix entries

### B01 — `model.train(True)` restored after PGD craft

- **Commit:** `da5d559`
- **File touched:** `src/adversarial_reasoning_training/trainer/adv_trainer.py`
- **Test added:** `tests/test_adv_trainer_train_mode.py::test_outer_step_runs_loss_fn_with_model_in_training_mode`
- **What changed:** `_outer_step` snapshots `self.model.training`, switches to eval for the PGD craft, then restores the prior mode inside a `try/finally`. Pre-fix the eval mode leaked into the outer forward+backward for the entire `fit()` run.
- **Behaviour preservation:** Models with dropout p=0 (the family-default for the projector layer in all three VLMs covered by `configs/*.yaml`) see bit-identical training because eval vs train is observationally identical when p=0 and BN is frozen. Models with non-zero dropout now actually train under the documented regularization regime — this is the intended behaviour per `docs/PROJECT_REPORT.md`; the prior state was an inadvertent no-dropout regime, not a deliberate choice.
- **Verification:** Test instruments `loss_fn` to capture `model.training` at the moment the closure is invoked inside the real `_outer_step` (PGD is stubbed via monkey-patch on the module-level `run_inner_pgd` symbol). Pre-fix: captured `False`. Post-fix: captured `True`. Existing `tests/test_nan_skip.py` and `tests/test_amp_clip_order.py` continue to pass — the `try/finally` does not perturb the optimizer-step or NaN-skip paths.

### B02 — `setup_seed` honours `training.seed` YAML key

- **Commit:** `56eef60`
- **Files touched:** `src/adversarial_reasoning_training/cli/train.py`
- **Tests added:** 3 in `tests/test_cli_train_helpers.py` — `test_resolve_seed_uses_train_cfg_when_cli_unset`, `test_resolve_seed_cli_overrides_train_cfg`, `test_resolve_seed_missing_train_cfg_key_raises`.
- **What changed:** `--seed` argparse default flipped from `0` to `None`. New `_resolve_seed(args, train_cfg)` helper: CLI override beats YAML, falls back to `train_cfg["seed"]`. `setup_seed` moved to after `_load_and_validate_configs` so the validated YAML seed is reachable.
- **Behaviour preservation:** Pipeline runs (`scripts/run_pipeline.sh:609` always passes `--seed "$SEED"`) — unchanged. `art-train` invocations with explicit `--seed N` — unchanged. The only behaviour change is `art-train` *without* `--seed`: previously seed=0, now reads YAML — this is the intended contract per `cli/schema.py:32` where `seed` is in `TRAINING_REQUIRED_KEYS`.
- **Verification:** Helper tests cover both branches plus the missing-key error path. Existing `tests/test_cli_entry_points.py` (`--help` + missing-args paths) and `tests/test_seed.py` continue to pass.

### B03 — `weight_decay` and `betas` threaded from YAML into `OptimConfig`

- **Commit:** `4478531`
- **Files touched:** `src/adversarial_reasoning_training/cli/train.py`
- **Tests added:** 2 in `tests/test_cli_train_helpers.py` — `test_build_optim_and_schedule_threads_weight_decay_and_betas`, `test_build_optim_and_schedule_keeps_optimconfig_defaults_when_absent`.
- **What changed:** `_build_optim_and_schedule` now reads `weight_decay` and `betas` from `train_cfg` (with the `OptimConfig` dataclass defaults as fallbacks) and passes them to `OptimConfig`. Mirrors the pattern already used by `gates/T1_clean.py:_build_t1_optimization`.
- **Behaviour preservation:** Configs that omit both keys keep the prior optimizer (weight_decay=0.0, betas=(0.9, 0.999)) — verified by the second test. Configs that set either key now get the value they declared. Before fixing, check `configs/training*.yaml` and `configs/ablations/*.yaml` for any active `weight_decay:` / `betas:` lines — if present, the next adv-FT run will train under a *different* optimizer than past runs (the YAML value, not the dataclass default). This is the intended contract per `cli/schema.py:79-80`.
- **Verification:** Direct optimizer-state assertions on `param_groups[*]["weight_decay"]` and `optimizer.defaults["betas"]`.

### B04 — T1 clean-FT NaN-loss guard

- **Commit:** `f61f802`
- **Files touched:** `src/adversarial_reasoning_training/gates/T1_clean.py`
- **Test added:** `tests/test_t1_nan_skip.py::test_t1_train_step_skips_nan_loss_and_preserves_weights`.
- **What changed:** `_t1_train_step` adds `torch.isfinite(loss).item()` guard before `(loss/grad_accum).backward()`. Non-finite loss → `optimizer.zero_grad(set_to_none=True)` + early return with `loss_val=nan` and `micro=0`. Mirrors `adv_trainer.py:_run_epoch` lines 460-462.
- **Behaviour preservation:** Finite-loss batches reach `backward()` and the existing accumulation/step flow unchanged — the guard is a strict subset of the prior code path. Only behavioural change: a NaN micro-batch no longer poisons subsequent steps; instead it is dropped (matching `task_ce`'s own documented expectation that "the trainer's finite-loss guard skips" the degenerate case).
- **Verification:** Test constructs a degenerate batch (all-zero `task_mask` → `task_ce` returns `sum/0 = nan` per its existing contract), invokes `_t1_train_step` directly, asserts (a) returned `loss_val` is NaN, (b) model weights bit-identical before/after, (c) step counter unchanged, (d) micro counter reset to 0.

### B05 — Directional `significant_bh` in T3 robust gate

- **Commit:** `fe4be21`
- **Files touched:** `src/adversarial_reasoning_training/gates/T3_robust.py`
- **Tests added:** 2 in `tests/test_gates.py` — `test_t3_fails_when_defense_significantly_degrades_metrics`, `test_t3_directional_check_counts_only_positive_deltas`.
- **What changed:** `_apply_bh_fdr` now sets `per_metric[key]["significant_bh"] = bool(rej and delta_mean >= 0)`. The two-sided Wilcoxon stays so the published `p_value` field reads as the familiar symmetric test of "any change", but a metric only counts toward `min_significant_metrics` when the directional improvement check also holds. All four T3 metrics are "higher = better" similarity scores, so the `>= 0` convention is uniform across the family.
- **Behaviour preservation:** **This is a semantic change to a published-results gate** — operator approved this fix explicitly before the commit. Pre-fix T3.json entries from past runs remain readable; only the boolean flag flips for the "significant but degraded" case, which the gate previously passed incorrectly. The existing `test_t3_passes_with_clear_robustness_gain` continues to pass (the clear-improvement fixture's deltas are all positive).
- **Verification:** The new `test_t3_fails_when_defense_significantly_degrades_metrics` reproduces the exact BUG_HUNT scenario: defense drops tool_name_acc / args_iou / answer_em by 0.4 each (significant under Wilcoxon at n=30) while keeping `traj_edit_distance` inside the `-min_traj_edit_delta` bound. Pre-fix: `passed=True` (the gate's semantic was inverted). Post-fix: `passed=False`, and each degraded metric is flagged `significant_bh=False`.

### B10 — Type-check `eps` in `validate_eps_schedule`

- **Commit:** `(see `git log` — second-batch ordering)`
- **File touched:** `src/adversarial_reasoning_training/attacks/inner_pgd.py`
- **Tests added:** 3 in `tests/test_epsilon_schedule_config.py` — `test_validate_rejects_non_numeric_eps`, `test_validate_accepts_int_eps`, `test_validate_rejects_bool_eps`.
- **What changed:** `validate_eps_schedule` now rejects non-numeric `eps` values at startup. A quoted `eps: "0.01"` previously passed shape validation and then crashed mid-attack with `unsupported operand type(s) for *: 'str' and 'float'` inside `PGDAttack` when computing `alpha = eps * alpha_ratio`. `bool` is also explicitly rejected because YAML `eps: true` coerces to `True` which is an `int` subclass and would silently mean `1.0 ε`.
- **Behaviour preservation:** Well-formed schedules (int or float eps) pass unchanged. The only new failure mode is the early surface of operator typing mistakes — the alternative was a mid-epoch H200 crash.
- **Verification:** Parametrised tests cover string ("0.01"), int (0), bool (True). The original three (`accepts_well_formed`, `accepts_none_or_empty`, `rejects_typo_in_epoch_range_key`) continue to pass.

### B11 — Allow `warmup_pct=0` to disable warmup

- **Commit:** `(see `git log`)`
- **File touched:** `src/adversarial_reasoning_training/trainer/optim.py`
- **Tests added:** 2 in `tests/test_optim.py` — `test_build_scheduler_warmup_pct_zero_starts_at_base_lr`, `test_build_scheduler_warmup_pct_nonzero_still_starts_at_zero`.
- **What changed:** `warmup_steps = max(1, math.ceil(...))` → `max(0, math.ceil(...))`. The `max(1, ...)` clamp meant `warmup_pct=0` produced `warmup_steps=1` so step 0 always trained at lr=0; operators had no way to opt out of warmup.
- **Behaviour preservation:** When `warmup_pct > 0`, `warmup_steps` is unchanged (the ceil result is already ≥ 1 for any positive product). Only the `warmup_pct=0` opt-out path changes — step 0 now returns the decay function at progress=0, which equals `base_lr` for constant / cosine schedules.
- **Verification:** Five pre-existing scheduler tests (cosine endpoints, linear, constant, cosine midpoint, the cosine warmup walk-through) continue to pass; new tests cover both the disable path and the unchanged warmup behaviour.

### B09 — Canonicalise `pi_rads` to `int` regardless of CSV spelling

- **Commit:** `(see `git log`)`
- **File touched:** `src/adversarial_reasoning_training/gold/oracle.py`
- **Tests added:** 1 parametrised test (4 spellings) in `tests/test_gold_extra.py` — `test_load_metadata_csv_pi_rads_canonical_int`.
- **What changed:** `_coerce_metadata` previously used a single ternary `float(md[k]) if "." in str(md[k]) or k != "pi_rads" else int(md[k])` keyed on the literal dot. For `pi_rads="3.0"` the dot routed to `float`, producing `3.0` where downstream JSON serialisation, template selection assertions, and metadata equality checks expect an `int`. Replaced with a branch: `pi_rads` → `int(float(md[k]))`, others → `float(md[k])`.
- **Behaviour preservation:** `int(float("3"))` == `int("3")` == 3, so the integer-string path is bit-identical. The only behavioural delta is dotted-float canonicalising int→int instead of int→float, which is the fix.
- **Verification:** Parametrised over `"3"`, `"3.0"`, `"3.00"`, `" 3 "` — all yield `int 3`. All four other `load_metadata_csv_*` tests continue to pass.

### B08 — Explicit `weights_only=False` in `load_checkpoint`

- **Commit:** `(see `git log`)`
- **File touched:** `src/adversarial_reasoning_training/trainer/ckpt.py`
- **Test added:** `tests/test_ckpt_rotation.py::test_load_checkpoint_passes_weights_only_false`.
- **What changed:** `torch.load(path, map_location=...)` → `torch.load(path, map_location=..., weights_only=False)`. The PyTorch 2.4 release emitted a FutureWarning on this default; 2.6 flipped it to `True`. Our payload contains the optimizer state_dict whose nested objects the safe loader rejects, so the next torch upgrade would silently break resume mid-run. The two existing `scripts/` loaders already pass the kwarg explicitly; this brings the trainer's loader into line.
- **Behaviour preservation:** PT < 2.6 default for this call shape was already `False`. PT ≥ 2.6 would have raised at load time; we now stay forward-compatible. Trusted-local-checkpoint loading is unchanged.
- **Verification:** Spy via `monkeypatch.setattr` on `torch.load` captures kwargs and asserts `weights_only is False`. Other three rotation tests continue to pass.

### B07 — Drop in-process `PYTHONHASHSEED` assignment

- **Commit:** `(see `git log`)`
- **Files touched:** `src/adversarial_reasoning_training/utils/seed.py`, `tests/test_seed.py`
- **Test modified:** `test_seed_everything_sets_pythonhashseed` (which pinned the buggy no-op) → `test_seed_everything_does_not_touch_pythonhashseed` (regression test that the misleading line cannot return).
- **What changed:** Removed `os.environ["PYTHONHASHSEED"] = str(seed)`. Python's hash randomization is locked in at interpreter startup, so the in-process assignment was a no-op that created false confidence in hash-stable dict / set iteration. Docstring now points operators to pin it via the shell launcher if needed.
- **Behaviour preservation:** The four RNG-determinism tests (Python, NumPy, torch CPU, cuDNN flags) continue to pass — actual seeding is untouched. The companion `os` import in `seed.py` is removed (orphaned by the line removal); `os` is retained in the test module for `os.environ` introspection.
- **Verification:** Suite full pass.

### B06 — T0 verdict is now freeze-aware per role

- **Commit:** `(see `git log`)`
- **File touched:** `src/adversarial_reasoning_training/gates/T0_env.py`
- **Tests added:** 5 in new `tests/test_t0_verdict.py` — covering trainable-with-zero-grad (fails), frozen-with-zero-grad (passes), all-zero-with-all-trainable (fails — pre-fix legacy path), `_trainable_roles` helper covering one-frozen and all-frozen cases.
- **What changed:** `_evaluate_t0_verdict` previously failed only when `_RoleGradNorms.all_zero` held (i.e., vit AND projector AND lm all had zero grad). T0's stated purpose is to catch "the freeze strategy did not accidentally disconnect a subgraph", but a real regression typically severs ONE role's grads while the other two still flow — so the gate silently passed exactly the failure mode it was meant to catch. New `_trainable_roles(model)` helper probes `requires_grad` against the existing role-pattern sets and returns the frozenset of role names with at least one trainable param. The verdict now fails when any trainable role's grad norm is exactly zero, naming the disconnected role in the notes. Intentionally frozen roles are exempt.
- **Behaviour preservation:** Pre-fix all-zero-fail mode is a strict subset of the new per-role check (the legacy-path test confirms). The behavioural change is the intended fix — single-role severance now fails the gate. Removed the now-orphaned `_RoleGradNorms.all_zero` property.
- **Verification:** Five new unit tests, all pass; existing public-API import test continues to pass.

---

## Out of scope

- **Pre-existing test failures retained:**
  - `tests/test_ablation_schemas.py::test_eps_reverse_inverts_undefended_curriculum`
  - `tests/test_ablation_schemas.py::test_loss_axis_actually_changes_defense`
  Both originate from the dirty branch state's `configs/defenses.yaml` ↔ `configs/ablations/*.yaml` drift introduced by the prior `42174f4 fix(training): default defense oaat, not pgd_at` commit, **not** by any Phase 2 fix. Per the mandate's "don't widen scope" clause, surfaced here rather than corrected. Reproducible on `HEAD~11` (before any of the 11 fixes shipped) — verified by inspection of the dirty file list in the initial git status.

---

## End state

Branch: `refactor/clean-code-bounded-funcs`, ahead of `origin/refactor/clean-code-bounded-funcs` by 13 commits (2 prior + 11 fix commits). Files still uncommitted at end of Phase 3: the same 24 dirty files present at start of session (configs, docs, scripts unrelated to any approved finding).
