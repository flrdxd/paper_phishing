"""
Quick Data Quality Test

This script tests the dataset quality checks to ensure synthetic
datasets are detected and rejected before training.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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

    # Create varied phishing emails
    phishing_templates = [
        "Urgent: Your account has been compromised. Click here to verify.",
        "Security alert: Unusual activity detected on your account.",
        "Immediate action required: Update your payment information.",
        "Warning: Your password will expire in 24 hours.",
        "Account suspension: Verify your identity to continue service."
    ]

    phishing_texts = []
    for i in range(100):
        template = phishing_templates[i % len(phishing_templates)]
        phishing_texts.append(f"{template} Reference: {i}")

    # Create varied legitimate emails
    legitimate_templates = [
        "Meeting reminder: Project discussion scheduled for tomorrow.",
        "Budget update: Q4 financial report is now available.",
        "Team announcement: New hire starting next week.",
        "Project status: Phase 1 completed successfully.",
        "Training opportunity: Register for upcoming workshop."
    ]

    legitimate_texts = []
    for i in range(100):
        template = legitimate_templates[i % len(legitimate_templates)]
        legitimate_texts.append(f"{template} ID: {i}")

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

    from utils.data_auditor import DataAuditor

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
            logger.error("✗ TEST FAILED: Synthetic dataset not detected")
    except Exception as e:
        logger.error(f"✗ TEST FAILED: Exception during audit: {e}")

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
            logger.warning(f"⚠️  TEST WARNING: Realistic dataset has {results['risk_level']} risk")
    except Exception as e:
        logger.error(f"✗ TEST FAILED: Exception during audit: {e}")


def test_preprocessing_quality_checks():
    """Test preprocessing quality checks."""
    logger.info("\n" + "="*60)
    logger.info("TESTING PREPROCESSING QUALITY CHECKS")
    logger.info("="*60)

    from data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor(max_features=100, random_state=42)

    # Test 1: Synthetic dataset (should RAISE ERROR)
    logger.info("\nTest 1: Synthetic dataset (should raise error)")
    synthetic_df = create_synthetic_dataset()
    try:
        preprocessor.preprocess_dataframe(synthetic_df)
        logger.error("✗ TEST FAILED: Synthetic dataset should have been rejected")
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
        logger.error(f"✗ TEST FAILED: Realistic dataset should have been accepted")
        logger.error(f"  Error: {e}")


def test_cached_dataset_detection():
    """Test detection of cached synthetic datasets."""
    logger.info("\n" + "="*60)
    logger.info("TESTING CACHED SYNTHETIC DATASET DETECTION")
    logger.info("="*60)

    from data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor(max_features=100, random_state=42)

    # Check if cached datasets exist and test them
    data_dir = 'data'
    phishing_file = os.path.join(data_dir, 'phishing_emails.csv')
    legitimate_file = os.path.join(data_dir, 'legitimate_emails.csv')

    if os.path.exists(phishing_file) and os.path.exists(legitimate_file):
        logger.info("Found cached dataset files")
        phishing_df = pd.read_csv(phishing_file)
        legitimate_df = pd.read_csv(legitimate_file)

        logger.info(f"Phishing samples: {len(phishing_df)}")
        logger.info(f"Legitimate samples: {len(legitimate_df)}")

        # Test if cached dataset is synthetic
        is_synthetic = preprocessor._is_synthetic_dataset(phishing_df, legitimate_df)

        if is_synthetic:
            logger.error("❌ CRITICAL: Cached dataset is SYNTHETIC!")
            logger.error("This will cause artificial accuracy.")
            logger.error("Delete cache files:")
            logger.error(f"  rm {phishing_file}")
            logger.error(f"  rm {legitimate_file}")
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


if __name__ == "__main__":
    main()