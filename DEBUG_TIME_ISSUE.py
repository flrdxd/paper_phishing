"""
DEBUG SCRIPT: Investigate 0.00 training time issue
"""

import time
import sys
import os

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

print("="*60)
print("DEBUG: Investigating 0.00 training time issue")
print("="*60)

# Test 1: Check time.time() behavior
print("\n[TEST 1] Time measurement check")
start = time.time()
print(f"Start time: {start}")
time.sleep(0.1)
end = time.time()
print(f"End time: {end}")
print(f"Elapsed: {end - start:.4f} seconds")

# Test 2: Try training MultinomialNB directly
print("\n[TEST 2] Direct MultinomialNB training")
preprocessor = DataPreprocessor(max_features=5000, random_state=42)

# Download and preprocess data (use cache)
try:
    df = preprocessor.download_dataset(force_download=False)
    df = preprocessor.preprocess_dataframe(df)
except Exception as e:
    print(f"ERROR loading data: {e}")
    sys.exit(1)

# Split data
X_train_ml, X_test_ml, y_train_ml, y_test_ml = preprocessor.split_for_ml(df, test_size=0.3)

print(f"Training data shape: {X_train_ml.shape}")
print(f"Training labels shape: {y_train_ml.shape}")

# Test direct training
nb_model = MultinomialNB(alpha=1.0, fit_prior=True)
start_time = time.time()
print(f"Training started at: {start_time}")

nb_model.fit(X_train_ml, y_train_ml)

end_time = time.time()
print(f"Training ended at: {end_time}")

elapsed = end_time - start_time
print(f"Training time: {elapsed:.4f} seconds ({elapsed/60:.2f} minutes)")

# Make predictions
predictions = nb_model.predict(X_test_ml)
accuracy = (predictions == y_test_ml).mean()
print(f"Test accuracy: {accuracy:.4f}")

# Test 3: Check if model is actually trained
print("\n[TEST 3] Model state check")
print(f"Model class log prior: {nb_model.class_log_prior_}")
print(f"Model feature log prob: {nb_model.feature_log_prob_.shape}")

# Test 4: Multiple training runs to see if time varies
print("\n[TEST 4] Multiple training runs")
times = []
for i in range(5):
    nb_test = MultinomialNB(alpha=1.0, fit_prior=True)
    start = time.time()
    nb_test.fit(X_train_ml, y_train_ml)
    end = time.time()
    elapsed = end - start
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.4f} seconds")

print(f"Time stats - Min: {min(times):.4f}s, Max: {max(times):.4f}s, Avg: {sum(times)/len(times):.4f}s")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)

if max(times) < 0.1:
    print("\n🚨 CRITICAL: All training times under 0.1 seconds!")
    print("This indicates a SERIOUS BUG:")
    print("1. Model not actually training (caching issue)")
    print("2. Time measurement failure")
    print("3. Data leakage causing instant training")
    print("4. Model loading instead of training")
else:
    print(f"\n✅ Training times look reasonable (avg: {sum(times)/len(times):.4f}s)")