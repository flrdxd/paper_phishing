"""
Quick Data Quality Test

This script tests the dataset quality checks to ensure synthetic
datasets are detected and rejected before training.
"""

import os

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_dataset():
    """Create a synthetic dataset for testing."""
    logger.info("Creating synthetic dataset for testing...")

    # Create 100 phishing emails (same template)
    phishing_texts = [
        f"Dear User, urgent notification from bank. Your account {i} is compromised."
        for i in range(100)
    ]

    # Create 100 legitimate emails (same template)
    legitimate_texts = [
        f"Hi there, meeting from yesterday about project {i}."
        for i in range(100)
    ]

    phishing_df = pd.DataFrame({
        'text': phishing_texts,
        'label': [1] * 100
    })

    legitimate_df = pd.DataFrame({
        'text': legitimate_texts,
        'label': [0] * 100
    })

    combined_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
    logger.info(f"Created synthetic dataset: {len(combined_df)} samples")
    return combined_df


def create_realistic_dataset():
    """Create a more realistic dataset for testing."""
    logger.info("Creating realistic dataset for testing...")

    phishing_subjects = [
        "bank profile", "cloud mailbox", "payroll portal", "shipping account",
        "tax document", "invoice approval", "password reset", "benefits login",
        "payment card", "security dashboard"
    ]
    phishing_actions = [
        "confirm access", "review alert", "restore service", "update records",
        "open notice", "validate request", "approve session", "check message",
        "complete review", "verify change"
    ]
    phishing_reasons = [
        "unusual sign in", "expired credential", "blocked transfer", "new device",
        "policy update", "returned payment", "pending case", "failed delivery",
        "account hold", "document release"
    ]

    phishing_texts = []
    for i in range(100):
        phishing_texts.append(
            f"Notice for {phishing_subjects[i % 10]} requires you to "
            f"{phishing_actions[(i // 10) % 10]} because of {phishing_reasons[(i * 3) % 10]}."
        )

    legitimate_subjects = [
        "roadmap review", "budget planning", "release notes", "team onboarding",
        "vendor renewal", "quarterly report", "support handoff", "design review",
        "legal feedback", "customer summary"
    ]
    legitimate_actions = [
        "share feedback", "join discussion", "review notes", "confirm attendance",
        "update timeline", "send questions", "approve minutes", "prepare slides",
        "check agenda", "summarize findings"
    ]
    legitimate_contexts = [
        "weekly meeting", "planning cycle", "internal project", "training session",
        "finance review", "client workshop", "engineering sync", "research update",
        "operations review", "handover call"
    ]

    legitimate_texts = []
    for i in range(100):
        legitimate_texts.append(
            f"Team note about {legitimate_subjects[i % 10]} asks everyone to "
            f"{legitimate_actions[(i // 10) % 10]} before the {legitimate_contexts[(i * 3) % 10]}."
        )

    phishing_df = pd.DataFrame({
        'text': phishing_texts,
        'label': [1] * 100
    })

    legitimate_df = pd.DataFrame({
        'text': legitimate_texts,
        'label': [0] * 100
    })

    combined_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
    logger.info(f"Created realistic dataset: {len(combined_df)} samples")
    return combined_df


def test_data_auditor():
    """Test the data auditor with synthetic and realistic datasets."""
    logger.info("="*60)
    logger.info("TESTING DATA AUDITOR")
    logger.info("="*60)

    from phishing_detection.utils.data_auditor import DataAuditor

    auditor = DataAuditor()

    # Test 1: Synthetic dataset (should FAIL)
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Synthetic Dataset (should detect issues)")
    logger.info("="*60)
    synthetic_df = create_synthetic_dataset()
    try:
        results = auditor.audit_dataset(synthetic_df, "Synthetic Dataset")
        logger.info(f"Risk Level: {results['risk_level']}")
        if results['risk_level'] == 'CRITICAL':
            logger.info("✓ TEST PASSED: Synthetic dataset correctly identified")
        else:
            raise AssertionError("Synthetic dataset not detected")
    except Exception as e:
        logger.error(f"✗ TEST FAILED: Exception during audit: {e}")
        raise

    # Test 2: Realistic dataset (should PASS or have LOW risk)
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Realistic Dataset (should pass or have low risk)")
    logger.info("="*60)
    realistic_df = create_realistic_dataset()
    try:
        results = auditor.audit_dataset(realistic_df, "Realistic Dataset")
        logger.info(f"Risk Level: {results['risk_level']}")
        if results['risk_level'] in ['LOW', 'MEDIUM']:
            logger.info("✓ TEST PASSED: Realistic dataset has acceptable risk")
        else:
            raise AssertionError(f"Realistic dataset has {results['risk_level']} risk")
    except Exception as e:
        logger.error(f"✗ TEST FAILED: Exception during audit: {e}")
        raise


def test_preprocessing_quality_checks():
    """Test preprocessing quality checks."""
    logger.info("\n" + "="*60)
    logger.info("TESTING PREPROCESSING QUALITY CHECKS")
    logger.info("="*60)

    from phishing_detection.data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor(max_features=100, random_state=42)

    # Test 1: Synthetic dataset (should RAISE ERROR)
    logger.info("\nTest 1: Synthetic dataset (should raise error)")
    synthetic_df = create_synthetic_dataset()
    try:
        preprocessor.preprocess_dataframe(synthetic_df)
        raise AssertionError("Synthetic dataset should have been rejected")
    except ValueError as e:
        logger.info(f"✓ TEST PASSED: Synthetic dataset correctly rejected")
        logger.info(f"  Error message: {e}")

    # Test 2: Realistic dataset (should PASS)
    logger.info("\nTest 2: Realistic dataset (should pass)")
    realistic_df = create_realistic_dataset()
    try:
        processed_df = preprocessor.preprocess_dataframe(realistic_df)
        logger.info(f"✓ TEST PASSED: Realistic dataset processed successfully")
        logger.info(f"  Processed {len(processed_df)} samples")
    except Exception as e:
        raise AssertionError(f"Realistic dataset should have been accepted: {e}") from e


def test_cached_dataset_detection():
    """Test detection of cached synthetic datasets."""
    logger.info("\n" + "="*60)
    logger.info("TESTING CACHED SYNTHETIC DATASET DETECTION")
    logger.info("="*60)

    from phishing_detection.utils.data_auditor import DataAuditor

    # Check if cached datasets exist and test them
    from phishing_detection.path_config import PATHS

    data_dir = PATHS['DATA_DIR']
    paper_file = os.path.join(data_dir, 'paper_phishing_dataset.csv')

    if os.path.exists(paper_file):
        logger.info("Found cached paper dataset file")
        cached_df = pd.read_csv(paper_file)

        logger.info(f"Cached samples: {len(cached_df)}")
        if 'label' in cached_df.columns:
            logger.info(f"Label distribution: {cached_df['label'].value_counts().to_dict()}")

        results = DataAuditor().audit_dataset(cached_df, "Cached dataset")
        is_synthetic = results['risk_level'] == 'CRITICAL'

        if is_synthetic:
            logger.error("❌ CRITICAL: Cached dataset is SYNTHETIC!")
            logger.error("This will cause artificial accuracy.")
            logger.error("Delete cache files:")
            logger.error(f"  rm {paper_file}")
        else:
            logger.info("✓ Cached dataset appears to be real")
    else:
        logger.info("No cached dataset files found")


def main():
    """Run all tests."""
    logger.info("Starting Data Quality Tests...")
    logger.info("These tests verify that synthetic/trivial datasets are detected")

    try:
        test_data_auditor()
        test_preprocessing_quality_checks()
        test_cached_dataset_detection()

        logger.info("\n" + "="*60)
        logger.info("ALL TESTS COMPLETED")
        logger.info("="*60)
        logger.info("\nIf tests passed, the quality checks are working correctly.")
        logger.info("If tests failed, review the error messages above.")

    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
