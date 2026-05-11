#!/usr/bin/env python3
"""
Minimal test script to identify the exact cause of bus error
Run this on the server to pinpoint the problematic component
"""

import sys
import os
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.path.join(project_root, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))

def test_step(description, test_func):
    """Test a single step and report results"""
    print(f"\n{'='*50}")
    print(f"TESTING: {description}")
    print(f"{'='*50}")
    try:
        result = test_func()
        print(f"✓ SUCCESS: {description}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {description}")
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def main():
    print("MINIMAL DEBUGGING SCRIPT FOR BUS ERROR")
    print("This will help identify the exact component causing the crash")
    print("="*60)

    # Test 1: Basic Python
    if not test_step("Basic Python operations", lambda:
        [1, 2, 3].index(2)):
        return 1

    # Test 2: Standard library imports
    if not test_step("Import os", lambda: __import__('os')):
        return 1

    if not test_step("Import sys", lambda: __import__('sys')):
        return 1

    # Test 3: NumPy
    if not test_step("Import numpy", lambda: __import__('numpy')):
        return 1

    # not test_step("Create numpy array", lambda: __import__('numpy').array([1, 2, 3])):
    #     return 1

    # Test 4: Pandas
    if not test_step("Import pandas", lambda: __import__('pandas')):
        return 1

    if not test_step("Create pandas DataFrame", lambda: __import__('pandas').DataFrame({'a': [1, 2, 3]})):
        return 1

    # Test 5: Scikit-learn
    if not test_step("Import sklearn", lambda: __import__('sklearn')):
        return 1

    # Test 6: NLTK
    if not test_step("Import nltk", lambda: __import__('nltk')):
        return 1

    # Test 7: BeautifulSoup
    if not test_step("Import bs4", lambda: __import__('bs4')):
        return 1

    # Test 8: PyTorch (MOST LIKELY CULPRIT)
    print("\n" + "="*50)
    print("TESTING: PyTorch (Most likely to cause bus error)")
    print("="*50)
    try:
        import torch
        print(f"✓ PyTorch imported successfully")
        print(f"  Version: {torch.__version__}")

        # Test tensor creation
        print(f"  Testing tensor creation...")
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"  ✓ Tensor created: {x}")

        # Test CUDA
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

        print("✓ PyTorch tests passed")
    except Exception as e:
        print(f"✗ PyTorch FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # Test 9: Transformers
    print("\n" + "="*50)
    print("TESTING: Transformers")
    print("="*50)
    try:
        import transformers
        print(f"✓ Transformers imported successfully")
        print(f"  Version: {transformers.__version__}")
    except Exception as e:
        print(f"✗ Transformers FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # Test 10: KaggleHub
    print("\n" + "="*50)
    print("TESTING: KaggleHub")
    print("="*50)
    try:
        import kagglehub
        print(f"✓ KaggleHub imported successfully")
    except Exception as e:
        print(f"✗ KaggleHub FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # Test 11: WordCloud
    print("\n" + "="*50)
    print("TESTING: WordCloud")
    print("="*50)
    try:
        from wordcloud import WordCloud
        print(f"✓ WordCloud imported successfully")
    except Exception as e:
        print(f"✗ WordCloud FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # Test 12: Project imports
    print("\n" + "="*50)
    print("TESTING: Project imports")
    print("="*50)
    try:
        from data_preprocessing import DataPreprocessor
        print(f"✓ DataPreprocessor imported")
    except Exception as e:
        print(f"✗ DataPreprocessor FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n" + "="*60)
        print(f"CRITICAL FAILURE: {type(e).__name__}: {e}")
        print("="*60)
        traceback.print_exc()
        sys.exit(2)