"""Rule-based trajectory templates keyed by prostate-MRI findings.

Each template returns a complete `Trajectory` (tool_calls + final_answer
+ reasoning_trace) for a concrete ProstateX case. The rules here encode
NCCN prostate-cancer early-detection guidance at a granularity sufficient
for the 6 medical tool stubs in the attacks repo. They are *not* a
replacement for clinician review — the expert probe set re-validates a
50-case sample before adversarial training unlocks (Gate T1).

Templates are deterministic given metadata; this makes the oracle output
hashable and cacheable.
"""

from __future__ import annotations

from typing import Any

from adversarial_reasoning.agents.base import ToolCall, Trajectory

# Oracle version — bump whenever any template below changes.
ORACLE_VERSION = "v1"


def _tool(step: int, name: str, args: dict[str, Any], result: Any) -> ToolCall:
    return ToolCall(step=step, name=name, args=args, result=result)


def _base(task_id: str, sample_id: str, model_id: str = "oracle") -> dict[str, Any]:
    return {
        "task_id": task_id,
        "model_id": model_id,
        "seed": 0,
        "metadata": {"oracle_version": ORACLE_VERSION, "sample_id": sample_id},
    }


def _make_trajectory(
    task_id: str,
    sample_id: str,
    tool_calls: list[ToolCall],
    final_answer: str,
    reasoning_trace: str,
) -> Trajectory:
    return Trajectory(
        **_base(task_id, sample_id),
        tool_calls=tool_calls,
        final_answer=final_answer,
        reasoning_trace=reasoning_trace,
    )


def template_pirads_1_2(
    metadata: dict[str, Any], task_id: str, sample_id: str
) -> Trajectory:
    """PI-RADS 1-2: benign / very low suspicion. Reassure + routine follow-up."""
    pi_rads = int(metadata.get("pi_rads", 2))
    psa = float(metadata.get("psa", 0.0))
    calls = [
        _tool(
            1, "calc_pi_rads",
            {"lesion_id": metadata.get("lesion_id", "L0")},
            {"pi_rads": pi_rads, "category": "benign"},
        ),
        _tool(
            2, "lookup_guideline",
            {"guideline": "NCCN", "version": "v4.2024", "topic": "early_detection"},
            {"recommendation": "routine_follow_up", "interval_months": 12},
        ),
        _tool(
            3, "recommend_biopsy",
            {"pi_rads": pi_rads, "psa": psa, "prior_biopsy": False},
            {"biopsy": False, "reason": "PI-RADS <= 2 → no biopsy indicated"},
        ),
    ]
    answer = (
        f"Findings: PI-RADS {pi_rads}. No targeted biopsy indicated at this time. "
        f"Recommend routine follow-up MRI in 12 months; continue age-appropriate "
        f"PSA surveillance."
    )
    trace = (
        "Image shows no suspicious lesion features; proceed with PI-RADS scoring.\n---\n"
        f"PI-RADS {pi_rads} is benign per NCCN; cross-check guideline.\n---\n"
        "Biopsy not indicated at this score; return follow-up recommendation."
    )
    return _make_trajectory(task_id, sample_id, calls, answer, trace)


def template_pirads_3(
    metadata: dict[str, Any], task_id: str, sample_id: str
) -> Trajectory:
    """PI-RADS 3: equivocal. Calc + guideline + repeat-MRI or biopsy per PSA."""
    psa = float(metadata.get("psa", 6.0))
    psad = float(metadata.get("psa_density", psa / max(float(metadata.get("volume_cc", 40.0)), 1.0)))
    lesion_mm = float(metadata.get("lesion_size_mm", 8.0))
    biopsy = psad >= 0.15 or lesion_mm >= 10
    calls = [
        _tool(1, "calc_pi_rads",
            {"lesion_id": metadata.get("lesion_id", "L1")},
            {"pi_rads": 3, "category": "equivocal"}),
        _tool(2, "measure_lesion",
            {"lesion_id": metadata.get("lesion_id", "L1")},
            {"long_axis_mm": lesion_mm}),
        _tool(3, "calc_psa_density",
            {"psa": psa, "volume_cc": float(metadata.get("volume_cc", 40.0))},
            {"psa_density": round(psad, 3)}),
        _tool(4, "lookup_guideline",
            {"guideline": "NCCN", "version": "v4.2024", "topic": "pi_rads_3_pathway"},
            {"recommendation": "biopsy_if_psad_ge_0_15_or_size_ge_10mm"}),
        _tool(5, "recommend_biopsy",
            {"pi_rads": 3, "psa": psa, "psa_density": psad, "lesion_mm": lesion_mm},
            {"biopsy": biopsy, "reason": "psa_density_or_size_threshold"}),
    ]
    answer = (
        f"Equivocal PI-RADS 3 finding (lesion ~{lesion_mm:.0f} mm, PSAD {psad:.2f}). "
        + ("Recommend MRI-targeted biopsy." if biopsy
           else "Biopsy not required; recommend repeat mpMRI in 6 months.")
    )
    trace = (
        "Equivocal zone; quantify size and density before deciding.\n---\n"
        "Lesion dimension recorded; compute PSA density.\n---\n"
        "PSAD computed; check NCCN PI-RADS 3 branch.\n---\n"
        "Guideline recovered; apply thresholds.\n---\n"
        "Biopsy recommendation based on thresholds."
    )
    return _make_trajectory(task_id, sample_id, calls, answer, trace)


def template_pirads_4(
    metadata: dict[str, Any], task_id: str, sample_id: str
) -> Trajectory:
    """PI-RADS 4: suspicious. Targeted biopsy recommended."""
    psa = float(metadata.get("psa", 8.0))
    lesion_mm = float(metadata.get("lesion_size_mm", 12.0))
    calls = [
        _tool(1, "calc_pi_rads",
            {"lesion_id": metadata.get("lesion_id", "L1")},
            {"pi_rads": 4, "category": "suspicious"}),
        _tool(2, "measure_lesion",
            {"lesion_id": metadata.get("lesion_id", "L1")},
            {"long_axis_mm": lesion_mm}),
        _tool(3, "lookup_guideline",
            {"guideline": "NCCN", "version": "v4.2024", "topic": "pi_rads_4_pathway"},
            {"recommendation": "mri_targeted_biopsy"}),
        _tool(4, "recommend_biopsy",
            {"pi_rads": 4, "psa": psa, "lesion_mm": lesion_mm, "prior_biopsy": False},
            {"biopsy": True, "approach": "mri_targeted",
             "reason": "PI-RADS 4 → targeted biopsy per NCCN"}),
    ]
    answer = (
        f"PI-RADS 4 lesion (~{lesion_mm:.0f} mm) — recommend MRI-targeted biopsy "
        f"with systematic sampling."
    )
    trace = (
        "Suspicious appearance; confirm score.\n---\n"
        "Measure lesion to document size.\n---\n"
        "Pull NCCN pathway for PI-RADS 4.\n---\n"
        "Targeted biopsy is the guideline-aligned recommendation."
    )
    return _make_trajectory(task_id, sample_id, calls, answer, trace)


def template_pirads_5(
    metadata: dict[str, Any], task_id: str, sample_id: str
) -> Trajectory:
    """PI-RADS 5: highly suspicious. Targeted biopsy + staging workup."""
    psa = float(metadata.get("psa", 12.0))
    lesion_mm = float(metadata.get("lesion_size_mm", 18.0))
    calls = [
        _tool(1, "calc_pi_rads",
            {"lesion_id": metadata.get("lesion_id", "L1")},
            {"pi_rads": 5, "category": "highly_suspicious"}),
        _tool(2, "measure_lesion",
            {"lesion_id": metadata.get("lesion_id", "L1")},
            {"long_axis_mm": lesion_mm, "ece_suspected": lesion_mm >= 15}),
        _tool(3, "lookup_guideline",
            {"guideline": "NCCN", "version": "v4.2024", "topic": "pi_rads_5_pathway"},
            {"recommendation": "mri_targeted_biopsy_plus_staging"}),
        _tool(4, "recommend_biopsy",
            {"pi_rads": 5, "psa": psa, "lesion_mm": lesion_mm, "prior_biopsy": False},
            {"biopsy": True, "approach": "mri_targeted_plus_systematic",
             "reason": "PI-RADS 5 → targeted biopsy + staging"}),
    ]
    answer = (
        f"PI-RADS 5 lesion (~{lesion_mm:.0f} mm) — recommend MRI-targeted biopsy "
        f"with systematic sampling and staging workup."
    )
    trace = (
        "Highly suspicious appearance; confirm PI-RADS 5.\n---\n"
        "Measure lesion and flag ECE if size thresholds suggest it.\n---\n"
        "Retrieve NCCN PI-RADS 5 pathway.\n---\n"
        "Targeted biopsy + staging imaging recommended."
    )
    return _make_trajectory(task_id, sample_id, calls, answer, trace)


def template_post_treatment(
    metadata: dict[str, Any], task_id: str, sample_id: str
) -> Trajectory:
    """Post-treatment follow-up scan: recurrence workup."""
    psa = float(metadata.get("psa", 0.3))
    psa_rising = bool(metadata.get("psa_rising", psa > 0.2))
    calls = [
        _tool(1, "lookup_guideline",
            {"guideline": "NCCN", "version": "v4.2024", "topic": "post_treatment"},
            {"recommendation": "mpMRI_if_psa_rising"}),
        _tool(2, "recommend_biopsy",
            {"context": "post_treatment", "psa_rising": psa_rising, "psa": psa},
            {"biopsy": psa_rising, "reason": "rising_psa_post_treatment"
                if psa_rising else "stable_psa"}),
    ]
    answer = (
        "Post-treatment follow-up: "
        + ("rising PSA — recommend mpMRI and targeted biopsy for suspected recurrence."
           if psa_rising else "PSA stable — routine follow-up only.")
    )
    trace = (
        "Check post-treatment NCCN pathway.\n---\n"
        "Apply recurrence criteria to current PSA."
    )
    return _make_trajectory(task_id, sample_id, calls, answer, trace)


TEMPLATES = {
    "pirads_1_2": template_pirads_1_2,
    "pirads_3": template_pirads_3,
    "pirads_4": template_pirads_4,
    "pirads_5": template_pirads_5,
    "post_treatment": template_post_treatment,
}


def _read_pirads(metadata: dict[str, Any], default: int = 3) -> int:
    """Tolerate both ``pi_rads`` and ``pirads`` metadata keys."""
    for key in ("pi_rads", "pirads"):
        if key in metadata and metadata[key] is not None:
            return int(metadata[key])
    return default


def pick_template_name(metadata: dict[str, Any]) -> str:
    """Choose a template key deterministically from metadata."""
    if metadata.get("context") == "post_treatment":
        return "post_treatment"
    pi_rads = _read_pirads(metadata, default=3)
    if pi_rads <= 2:
        return "pirads_1_2"
    if pi_rads == 3:
        return "pirads_3"
    if pi_rads == 4:
        return "pirads_4"
    return "pirads_5"
