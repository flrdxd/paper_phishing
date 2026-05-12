#!/bin/bash
#
# One-Click Setup and Run Script
# Verifica configuração, baixa dados reais e executa o pipeline
#

set -e  # Stop on error

echo "========================================"
echo "PHISHING DETECTION - AUTO SETUP & RUN"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Step 1: Check virtual environment
echo "Step 1: Checking virtual environment..."
if [ -d "venv" ]; then
    print_success "Virtual environment found"
    source venv/bin/activate
else
    print_error "Virtual environment not found"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    print_success "Virtual environment created"
fi

# Step 2: Install dependencies
echo ""
echo "Step 2: Installing/updating dependencies..."
pip install --quiet pandas numpy scikit-learn nltk beautifulsoup4 kaggle kagglehub joblib matplotlib seaborn
print_success "Dependencies installed"

# Step 3: Check Kaggle setup
echo ""
echo "Step 3: Checking Kaggle API setup..."
python check_kaggle_setup.py
KAGGLE_STATUS=$?

if [ $KAGGLE_STATUS -ne 0 ]; then
    echo ""
    print_error "Kaggle API is not properly configured"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/"
    echo "2. Create an account (free)"
    echo "3. Go to https://www.kaggle.com/settings"
    echo "4. Click 'Create New Token' in API section"
    echo "5. Download kaggle.json"
    echo "6. Run these commands:"
    echo ""
    echo "   mkdir -p ~/.kaggle"
    echo "   mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "Then run this script again."
    exit 1
fi

print_success "Kaggle API is properly configured"

# Step 4: Check data quality
echo ""
echo "Step 4: Checking data quality..."
if [ -f "data/phishing_emails.csv" ] && [ -f "data/legitimate_emails.csv" ]; then
    print_warning "Cached data found. Running quality checks..."
    python test_data_quality.py
    QUALITY_STATUS=$?

    if [ $QUALITY_STATUS -ne 0 ]; then
        echo ""
        print_warning "Quality checks failed. Data might be synthetic."
        read -p "Do you want to delete cached data and download fresh? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Deleting cached data..."
            rm -f data/phishing_emails.csv data/legitimate_emails.csv
            print_success "Cached data deleted"
        else
            print_error "Cannot proceed with potentially synthetic data"
            exit 1
        fi
    fi
else
    print_success "No cached data found. Will download fresh data."
fi

# Step 5: Choose pipeline
echo ""
echo "Step 5: Choose pipeline to run:"
echo "1) Scikit-learn only (recommended, faster, no GPU needed)"
echo "2) Complete pipeline with BERT/DistilBERT (slower, requires GPU)"
read -p "Enter choice (1 or 2): " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "Running scikit-learn only pipeline..."
        echo "This may take 10-30 minutes depending on your system."
        echo ""
        python src/main_sklearn_only.py
        ;;
    2)
        echo ""
        print_warning "Complete pipeline requires PyTorch and preferably a GPU"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Running complete pipeline..."
            echo "This may take 1-3 hours depending on your system."
            echo ""
            python src/main.py
        else
            echo "Cancelled. You can run the scikit-learn pipeline instead."
            exit 0
        fi
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Step 6: Show results
echo ""
echo "========================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/results_summary.txt"
echo "  - results/model_results.json"
echo "  - results/model_results.csv"
echo ""
echo "Plots saved to:"
echo "  - plots/*.png"
echo ""
echo "Models saved to:"
echo "  - models/*.pkl"
echo "  - models/*.pth (if transformers were trained)"
echo ""
echo "Logs saved to:"
echo "  - logs/*.log"
echo ""
print_success "All done! Check the results directory for outputs."

# Optional: Open results
if [ -f "results/results_summary.txt" ]; then
    echo ""
    read -p "Would you like to view the results summary? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        less results/results_summary.txt
    fi
fi