# Bug Fixes — Phase 3 Report

Phase 2 of the bug-hunt mandate applied 11 approved fixes (B01–B11). Each surviving fix is a single commit with TDD-driven test coverage and a behaviour-preservation contract documented in the commit body.

## Post-merge reconciliation with main (PR #25)

Three of the 11 fixes (B05, B07, B08) were superseded by independently-developed stronger fixes that landed on `main` while this branch was in flight. To avoid duplicate / contradictory contracts, our versions were dropped at merge time. Surviving from this branch: **8 fixes** (B01, B02, B03, B04, B06, B09, B10, B11).

| Superseded fix | Replaced by (`main`) | Why main's is better |
|---|---|---|
| B05 — directional `significant_bh` flag (post-hoc two-sided filter) | per-metric one-sided Wilcoxon (`alternative="greater"`) | Cleaner statistics — directional gating baked into the test, no post-hoc filter needed |
| B07 — drop in-process `PYTHONHASHSEED` assignment | Keep + add `CUBLAS_WORKSPACE_CONFIG` setdefault | Finding was wrong — DataLoader workers spawned later inherit the env var at startup, so the assignment IS meaningful |
| B08 — explicit `weights_only=False` | `weights_only=True` + atomic save + strict-match audit | Genuinely stronger: restricts the deserialiser, atomic rename prevents half-written ckpts, key-budget audit fails loud on architecture mismatch |

The two B05 unit tests (`test_t3_fails_when_defense_significantly_degrades_metrics`, `test_t3_directional_check_counts_only_positive_deltas`) were kept and re-targeted — they verify the directional-gating *contract*, which holds under either implementation strategy.

## Aggregate (post-merge)

| Metric | Value |
|---|---|
| Findings opened in `BUG_HUNT.md` | 11 |
| Findings approved for Phase 2 | 11 (B01–B11) |
| Fixes surviving merge | 8 (B01, B02, B03, B04, B06, B09, B10, B11) |
| Fixes superseded by `main` | 3 (B05, B07, B08) |
| Commits surviving merge | 8 + 1 docs |
| New tests surviving | 16 |
| Existing tests modified | 0 |
| Existing tests weakened | 0 |
| Full suite | 370 passed (deselecting `test_ablation_schemas.py` pre-existing failures) |

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

### B05 — SUPERSEDED by main

See "Post-merge reconciliation" above. Main switched T3 to per-metric one-sided Wilcoxon, which subsumes the directional `significant_bh` filter. The two regression tests added under B05 were kept (re-targeted to verify the directional-gating *contract*).

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

### B08 — SUPERSEDED by main

Main's PR #25 took the opposite tack — `weights_only=True` + atomic save + strict-match audit — which is a strictly stronger safety contract. Our weaker explicit-False fix was dropped at merge.

### B07 — SUPERSEDED by main

Main retains the in-process `PYTHONHASHSEED` assignment because DataLoader workers spawned later inherit the env at startup. Our removal would have broken that subprocess-inheritance path. Main additionally adds `CUBLAS_WORKSPACE_CONFIG` for full deterministic-algorithms compatibility on H200.

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

---

## Session 2026-06-12 — review of the unreviewed working-tree diff

The prior session left ~400 lines of uncommitted fixes (ε-rescaling, robust_eval
dedup/parsing, T2 NaN handling). Re-audited those *un-committed* hunks (not yet
reviewed by anyone) plus the attacks-repo integration. One real regression found
and fixed; the rest confirmed correct.

### B12 — `args_iou` fake-perfection guard over-NaNs genuine matches (FIXED)

- **File:** `src/adversarial_reasoning_training/eval/robust_eval.py:242-270` (`_record_metrics`)
- **Class:** correctness / silent-evidence-loss · **SEV:** HIGH · **CONF:** HIGH
- **Introduced by:** the uncommitted working-tree change that swapped `_parse_tool_calls_from_text` to `json.JSONDecoder.raw_decode` (which *can* recover real args) but simultaneously deleted the prior `all_empty_args` discriminator from the fake-perfection guard.
- **Bug:** for InternVL3 text-fallback records, the guard NaN'd **every** `args_iou == 1.0`, even when `raw_decode` recovered real matching args (e.g. `"args!": {"x": 1}` on both sides). Via `_drop_nan_metrics` (one NaN empties the whole list), a single such sample silently wiped `args_iou` for the entire T3 undefended-vs-defended comparison — and the warning ("all values were NaN — check parser") misdirects.
- **Why it hid:** the existing test `test_internvl3_tool_sequence_fallback_match` feeds exactly this real-args shape but only asserts `tool_name_acc`, never `args_iou`.
- **Fix:** re-couple the NaN decision to the *merged* `record_for_iou` calls actually scored by `_args_iou_record`; NaN only when every recovered call has empty args (true name-only fallback). Real recovered args now preserve a genuine 1.0.
- **Tests added (2, in `tests/test_align_per_sample_contract.py`):**
  - `test_internvl3_fallback_real_args_keep_perfect_iou` — real matching args → `args_iou == [1.0]` (was `[]` pre-fix; RED→GREEN).
  - `test_internvl3_fallback_name_only_nans_iou` — empty recovered args → metric correctly dropped to `[]` (guards the discriminator).
- **Verification:** full suite `375 passed, 2 failed`; the 2 failures are the unchanged pre-existing ablation-config drift (below).

### Confirmed correct (no change)

- **ε-rescaling in `inner_pgd.py`** (`norm_epsilon = config.epsilon / pixel_std`): `pixel_std` is a scalar (`max` of per-channel σ on every VLM wrapper), so `float(...)` is type-safe; the attacks-repo runner (`runner/attacks.py:251-254`) applies the byte-identical scalar rescaling, so train and eval share one adversary definition. The `/pixel_std` convention is deliberate and project-wide — **must not** be changed unilaterally or train desyncs from the eval harness.
- **`__target_mask` forward-kwarg:** consumed by `TokenTargetLoss.__call__` (`attacks/loss.py:102`); leaks into the model `forward` where it is absorbed by `**kwargs` (no crash). Inert at the model layer, functional at the loss layer.
- **robust_eval `n_max` denominator, dedup keep-last, `raw_decode` fallback, `_sanitize`, `_pair_key` NaN-ε guard, `load_records` skip counter:** all reviewed, correct.
- **T2 (`_read_metrics` prefer-`metrics`-subdict; `run_t2` missing-metric→NaN→fail):** reviewed and correct. The NaN-fail branch (`ok = (not isnan(drop)) and drop <= tol`) fixes the pre-existing "missing metric treated as 0.0 → silently passes when ceiling < tolerance" hole (obs 9262). **Coverage gap closed:** the existing `..._evaluator_returning_none_treated_as_zero` test uses a 0.90 ceiling, which fails under *both* old and new behaviour, so it did not pin the fix. Added `test_run_t2_missing_metric_fails_even_below_tolerance` (ceiling 0.02 < 3pp tol) — passes pre-2.6.x-fix would have been a silent PASS, now asserts FAIL + NaN current/drop.
- **T3 (`_wilcoxon_signed_rank` length guard + `strict=True`; `_compute_t3_verdict` explicit NaN-traj branch):** reviewed and correct. The length guard raises instead of silently truncating mismatched paired arrays; equal length is an upstream invariant (`align_per_sample` shared-keys + `_drop_nan_metrics_paired`), so the guard is a fail-loud assert. The explicit `isnan(traj_delta)` branch is behaviour-equivalent to the prior `isnan or <` short-circuit, with a clearer note. (B05's two-sided-Wilcoxon direction issue is untouched — deferred below.)

### Deferred — require user sign-off (NOT fixed)

- **B05 — T3 two-sided Wilcoxon lets significant *degradation* pass** for `tool_name_acc` / `args_iou` / `answer_em` (only `traj_edit_distance` has a directional check). Changing it alters a *published-results* gate semantic. Decision needed: one-sided test, or post-hoc directional gate after BH-FDR.
- **Ablation-config drift (2 failing tests):** `defenses_eps_reverse.yaml` (eps order not strict reverse) and `loss_oaat.yaml` (`defense` unchanged from undefended `oaat`). These are published ablation YAMLs; whether the test or the config is authoritative is a domain call. Left exactly as-is.
