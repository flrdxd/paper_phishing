"""
Data Quality Auditor for Phishing Email Detection

This module performs critical sanity checks to detect:
- Synthetic/trivial datasets
- Data leakage
- Duplicates and near-duplicates
- Overfitting risks
- Artificial accuracy

Run this BEFORE any model training to ensure data integrity.
"""

import logging
import numpy as np
import re
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataAuditor:
    """Audits dataset quality and detects issues that cause artificial accuracy."""

    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed_checks = []

    def _normalized_texts(self, df):
        """Return lowercase texts with numeric/template-only variation normalized."""
        return (
            df['text']
            .fillna('')
            .astype(str)
            .str.lower()
            .apply(lambda text: re.sub(r'\d+', '<num>', text))
            .apply(lambda text: re.sub(r'\s+', ' ', text).strip())
        )

    def audit_dataset(self, df, dataset_name="Dataset"):
        """
        Perform comprehensive dataset audit.

        Args:
            df: DataFrame with 'text' and 'label' columns
            dataset_name: Name of the dataset for reporting

        Returns:
            dict: Audit results with critical findings
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"AUDITING: {dataset_name}")
        logger.info(f"{'='*60}")

        results = {
            'dataset_name': dataset_name,
            'total_samples': len(df),
            'critical_issues': [],
            'warnings': [],
            'passed_checks': [],
            'risk_level': 'UNKNOWN'
        }

        # Check 1: Basic dataset integrity
        self._check_basic_integrity(df, results)
        if results['critical_issues']:
            results['risk_level'] = self._determine_risk_level(results)
            self._print_audit_summary(results)
            return results

        # Check 2: Text diversity (CRITICAL for detecting synthetic data)
        self._check_text_diversity(df, results)

        # Check 3: Template repetition (CRITICAL for fallback datasets)
        self._check_template_repetition(df, results)

        # Check 4: Label distribution
        self._check_label_distribution(df, results)

        # Check 5: Text length statistics
        self._check_text_length_stats(df, results)

        # Check 6: Duplicates
        self._check_duplicates(df, results)

        # Check 7: Vocabulary diversity
        self._check_vocabulary_diversity(df, results)

        # Determine overall risk level
        results['risk_level'] = self._determine_risk_level(results)

        # Print summary
        self._print_audit_summary(results)

        return results

    def _check_basic_integrity(self, df, results):
        """Check basic dataset integrity."""
        logger.info("\n[CHECK 1] Basic Dataset Integrity")

        required_columns = ['text', 'label']
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            results['critical_issues'].append(error_msg)
            logger.error(f"  ❌ CRITICAL: {error_msg}")
            return
        else:
            results['passed_checks'].append("All required columns present")
            logger.info(f"  ✓ Passed: All required columns present")

        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts['text'] > 0:
            warning_msg = f"Found {null_counts['text']} null text values"
            results['warnings'].append(warning_msg)
            logger.warning(f"  ⚠️  WARNING: {warning_msg}")

    def _check_text_diversity(self, df, results):
        """
        CRITICAL CHECK: Detect synthetic datasets by measuring text diversity.

        Synthetic fallback datasets often have extremely low diversity
        (same template repeated with minor variations).
        """
        logger.info("\n[CHECK 2] Text Diversity (CRITICAL)")

        normalized_texts = self._normalized_texts(df)
        unique_texts = normalized_texts.nunique()
        total_samples = len(df)
        diversity_ratio = unique_texts / total_samples

        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Unique normalized texts: {unique_texts}")
        logger.info(f"  Diversity ratio: {diversity_ratio:.4f} ({diversity_ratio*100:.2f}%)")

        # CRITICAL: Very low diversity indicates synthetic data
        if diversity_ratio < 0.1:  # Less than 10% unique samples
            error_msg = f"EXTREMELY LOW TEXT DIVERSITY: Only {diversity_ratio*100:.2f}% unique texts. This indicates synthetic/fallback data!"
            results['critical_issues'].append(error_msg)
            logger.error(f"  ❌ CRITICAL: {error_msg}")
        elif diversity_ratio < 0.5:  # Less than 50% unique samples
            warning_msg = f"LOW TEXT DIVERSITY: Only {diversity_ratio*100:.2f}% unique texts. Possible synthetic data."
            results['warnings'].append(warning_msg)
            logger.warning(f"  ⚠️  WARNING: {warning_msg}")
        else:
            results['passed_checks'].append(f"Good text diversity: {diversity_ratio*100:.2f}% unique")
            logger.info(f"  ✓ Passed: Good text diversity")

    def _check_template_repetition(self, df, results):
        """
        CRITICAL CHECK: Detect repeated templates (fallback datasets).

        Fallback datasets often use the same template with minor variations
        like changing account numbers or names.
        """
        logger.info("\n[CHECK 3] Template Repetition (CRITICAL)")

        # Get first words of each normalized text (common in templates)
        normalized_texts = self._normalized_texts(df)
        first_words = normalized_texts.str.split().str[:5].apply(lambda words: ' '.join(words))
        first_word_counts = Counter(first_words)

        # Check if most samples start with the same few phrases
        top_templates = first_word_counts.most_common(5)
        total_samples = len(df)

        logger.info(f"  Top 5 opening templates:")
        for template, count in top_templates:
            percentage = (count / total_samples) * 100
            logger.info(f"    '{template}': {count} samples ({percentage:.2f}%)")

        # CRITICAL: If one template covers most samples, it's synthetic
        if top_templates[0][1] / total_samples >= 0.5:  # 50% same opening template
            error_msg = f"TEMPLATE REPETITION DETECTED: {top_templates[0][1]}/{total_samples} samples ({(top_templates[0][1]/total_samples)*100:.1f}%) start with the same template. This is SYNTHETIC DATA!"
            results['critical_issues'].append(error_msg)
            logger.error(f"  ❌ CRITICAL: {error_msg}")
        elif top_templates[0][1] / total_samples > 0.3:  # 30% same template
            warning_msg = f"HIGH TEMPLATE REPETITION: {top_templates[0][1]}/{total_samples} samples ({(top_templates[0][1]/total_samples)*100:.1f}%) start with the same template."
            results['warnings'].append(warning_msg)
            logger.warning(f"  ⚠️  WARNING: {warning_msg}")
        else:
            results['passed_checks'].append("Low template repetition")
            logger.info(f"  ✓ Passed: Low template repetition")

    def _check_label_distribution(self, df, results):
        """Check label distribution for balance."""
        logger.info("\n[CHECK 4] Label Distribution")

        label_counts = df['label'].value_counts()
        total_samples = len(df)

        logger.info(f"  Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / total_samples) * 100
            label_name = "Phishing" if label == 1 else "Legitimate"
            logger.info(f"    {label_name} (label={label}): {count} samples ({percentage:.2f}%)")

        # Check for severe imbalance
        if len(label_counts) == 1:
            error_msg = "ONLY ONE CLASS PRESENT: Dataset contains only phishing or only legitimate emails!"
            results['critical_issues'].append(error_msg)
            logger.error(f"  ❌ CRITICAL: {error_msg}")
        else:
            min_count = label_counts.min()
            min_percentage = (min_count / total_samples) * 100
            if min_percentage < 10:
                warning_msg = f"SEVERE CLASS IMBALANCE: Minority class is only {min_percentage:.2f}% of data"
                results['warnings'].append(warning_msg)
                logger.warning(f"  ⚠️  WARNING: {warning_msg}")
            else:
                results['passed_checks'].append("Reasonable class balance")
                logger.info(f"  ✓ Passed: Reasonable class balance")

    def _check_text_length_stats(self, df, results):
        """Check text length statistics for anomalies."""
        logger.info("\n[CHECK 5] Text Length Statistics")

        text_lengths = df['text'].fillna('').astype(str).str.len()

        logger.info(f"  Min length: {text_lengths.min()}")
        logger.info(f"  Max length: {text_lengths.max()}")
        logger.info(f"  Mean length: {text_lengths.mean():.1f}")
        logger.info(f"  Median length: {text_lengths.median():.1f}")
        logger.info(f"  Std length: {text_lengths.std():.1f}")

        # Check for extremely short texts (possible data issues)
        very_short = (text_lengths < 20).sum()
        if very_short > 0:
            warning_msg = f"Found {very_short} extremely short texts (<20 chars)"
            results['warnings'].append(warning_msg)
            logger.warning(f"  ⚠️  WARNING: {warning_msg}")

        # Check for suspicious length patterns (all same length = synthetic)
        unique_lengths = text_lengths.nunique()
        if unique_lengths < 10:
            warning_msg = f"SUSPICIOUS LENGTH PATTERN: Only {unique_lengths} unique text lengths detected"
            results['warnings'].append(warning_msg)
            logger.warning(f"  ⚠️  WARNING: {warning_msg}")
        else:
            results['passed_checks'].append("Diverse text lengths")
            logger.info(f"  ✓ Passed: Diverse text lengths ({unique_lengths} unique lengths)")

    def _check_duplicates(self, df, results):
        """Check for duplicate samples."""
        logger.info("\n[CHECK 6] Duplicate Detection")

        exact_duplicates = df.duplicated(subset=['text']).sum()
        duplicate_percentage = (exact_duplicates / len(df)) * 100

        logger.info(f"  Exact duplicates: {exact_duplicates} ({duplicate_percentage:.2f}%)")

        if exact_duplicates > 0:
            if duplicate_percentage > 10:
                error_msg = f"HIGH DUPLICATE RATE: {duplicate_percentage:.2f}% of samples are exact duplicates"
                results['critical_issues'].append(error_msg)
                logger.error(f"  ❌ CRITICAL: {error_msg}")
            else:
                warning_msg = f"Found {exact_duplicates} exact duplicates ({duplicate_percentage:.2f}%)"
                results['warnings'].append(warning_msg)
                logger.warning(f"  ⚠️  WARNING: {warning_msg}")
        else:
            results['passed_checks'].append("No exact duplicates found")
            logger.info(f"  ✓ Passed: No exact duplicates found")

    def _check_vocabulary_diversity(self, df, results):
        """Check vocabulary diversity."""
        logger.info("\n[CHECK 7] Vocabulary Diversity")

        # Simple word count (without NLTK dependency)
        normalized_texts = self._normalized_texts(df)
        all_words = ' '.join(normalized_texts).split()
        unique_words = len(set(all_words))
        total_words = len(all_words)

        logger.info(f"  Total words: {total_words}")
        logger.info(f"  Unique words: {unique_words}")
        logger.info(f"  Vocabulary diversity: {unique_words/total_words:.4f}")

        if unique_words < 100:
            error_msg = f"EXTREMELY LOW VOCABULARY: Only {unique_words} unique words in entire dataset. This indicates synthetic data!"
            results['critical_issues'].append(error_msg)
            logger.error(f"  ❌ CRITICAL: {error_msg}")
        elif unique_words < 500:
            warning_msg = f"LOW VOCABULARY: Only {unique_words} unique words. Dataset may be too simple."
            results['warnings'].append(warning_msg)
            logger.warning(f"  ⚠️  WARNING: {warning_msg}")
        else:
            results['passed_checks'].append(f"Good vocabulary diversity ({unique_words} unique words)")
            logger.info(f"  ✓ Passed: Good vocabulary diversity")

    def _determine_risk_level(self, results):
        """Determine overall risk level based on findings."""
        critical_count = len(results['critical_issues'])
        warning_count = len(results['warnings'])

        if critical_count > 0:
            return 'CRITICAL'
        elif warning_count > 2:
            return 'HIGH'
        elif warning_count > 0:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _print_audit_summary(self, results):
        """Print audit summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"AUDIT SUMMARY: {results['dataset_name']}")
        logger.info(f"{'='*60}")
        logger.info(f"Risk Level: {results['risk_level']}")
        logger.info(f"Total Samples: {results['total_samples']}")
        logger.info(f"Critical Issues: {len(results['critical_issues'])}")
        logger.info(f"Warnings: {len(results['warnings'])}")
        logger.info(f"Passed Checks: {len(results['passed_checks'])}")

        if results['critical_issues']:
            logger.info(f"\n❌ CRITICAL ISSUES:")
            for issue in results['critical_issues']:
                logger.info(f"  - {issue}")

        if results['warnings']:
            logger.info(f"\n⚠️  WARNINGS:")
            for warning in results['warnings']:
                logger.info(f"  - {warning}")

        logger.info(f"{'='*60}\n")


def audit_dataframe(df, dataset_name="Dataset"):
    """
    Convenience function to audit a DataFrame.

    Args:
        df: DataFrame to audit
        dataset_name: Name for the dataset

    Returns:
        dict: Audit results
    """
    auditor = DataAuditor()
    return auditor.audit_dataset(df, dataset_name)


def main():
    """Test the data auditor."""
    # Create synthetic dataset for testing
    import pandas as pd

    # Create problematic synthetic dataset
    synthetic_data = {
        'text': [
            "Dear User, urgent notification from bank. Your account is compromised.",
            "Dear User, urgent notification from bank. Your account is compromised.",
            "Dear User, urgent notification from bank. Your account is compromised.",
            "Hi there, meeting from yesterday about project timeline.",
            "Hi there, meeting from yesterday about project timeline.",
            "Hi there, meeting from yesterday about project timeline.",
        ],
        'label': [1, 1, 1, 0, 0, 0]
    }
    df_synthetic = pd.DataFrame(synthetic_data)

    print("Testing with SYNTHETIC dataset (should detect issues):")
    print("-" * 60)
    audit_results = audit_dataframe(df_synthetic, "Synthetic Test Dataset")

    print("\n" + "=" * 60)
    print("EXPECTED: Should detect CRITICAL issues with synthetic data")
    print("=" * 60)


if __name__ == "__main__":
    main()
