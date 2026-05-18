"""Orchestrator for the paper-v1 research pipeline."""

from __future__ import annotations

import logging

from phishing_detection.experiments.h1_meajor import run_h1_meajor
from phishing_detection.experiments.h2_url_meta import run_h2_url_meta
from phishing_detection.experiments.h3_fusion import run_h3_fusion
from phishing_detection.experiments.h4_encoders import run_h4_compact_encoders
from phishing_detection.experiments.h5_robustness import run_h5_robustness
from phishing_detection.experiments.h6_base_rates import run_h6_base_rates


logger = logging.getLogger(__name__)


PROFILE_SAMPLE_SIZES = {
    "smoke": 1000,
    "full": None,
    "confirmatory": None,
}


def run_paper_v1(profile: str = "smoke", random_state: int = 42, force_download: bool = False) -> dict:
    """Run the implemented paper-v1 experiments in sequence."""
    if profile not in PROFILE_SAMPLE_SIZES:
        raise ValueError("profile must be one of: smoke, full, confirmatory")

    sample_size = PROFILE_SAMPLE_SIZES[profile]
    logger.info("=" * 60)
    logger.info("PAPER V1 EXPERIMENT SUITE: profile=%s", profile)
    logger.info("=" * 60)

    outputs = {
        "h1_random": run_h1_meajor(sample_size=sample_size, split="random", models="nb,logreg,svm", force_download=force_download, random_state=random_state),
        "h1_leave_one_source_out": run_h1_meajor(sample_size=sample_size, split="leave-one-source-out", models="nb,logreg,svm", force_download=False, random_state=random_state),
        "h2_random": run_h2_url_meta(sample_size=sample_size, split="random", force_download=False, random_state=random_state),
        "h2_leave_one_source_out": run_h2_url_meta(sample_size=sample_size, split="leave-one-source-out", force_download=False, random_state=random_state),
        "h3_random": run_h3_fusion(sample_size=sample_size, split="random", force_download=False, random_state=random_state),
        "h5_robustness": run_h5_robustness(sample_size=sample_size, force_download=False, random_state=random_state),
        "h6_base_rates": run_h6_base_rates(sample_size=sample_size, split="random", force_download=False, random_state=random_state),
    }
    if profile in {"full", "confirmatory"}:
        outputs["h4_compact_encoders"] = run_h4_compact_encoders(
            sample_size=sample_size,
            models="distilbert,minilm",
            epochs=3 if profile == "full" else 5,
            max_length=256,
            batch_size=8,
            force_download=False,
            random_state=random_state,
        )
    logger.info("Paper-v1 suite complete")
    return outputs
