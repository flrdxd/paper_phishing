# Server Setup Instructions

## Quick Setup for Paper Phishing Detection Project

### 1. Install Dependencies
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies with specific kaggle version
pip install -r requirements.txt
```

### 2. Configure Kaggle API
```bash
# Create kaggle directory
mkdir -p ~/.kaggle

# Create kaggle.json file with your credentials
cat > ~/.kaggle/kaggle.json << EOF
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_KAGGLE_API_KEY"
}
EOF

# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Get Kaggle Credentials
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json`
5. Copy the username and key to the file above

### 4. Verify Setup
```bash
# Test package imports
python3 -c "import sklearn; import bs4; import pandas; import numpy; print('✓ All packages working!')"

# Test Kaggle API
python3 -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); print('✓ Kaggle API working!')"

# Run the setup checker (may have some warnings but should work)
python check_kaggle_setup.py
```

### 5. Download Datasets
```bash
# Download phishing emails dataset
python3 -c "
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files('subhajournal/phishingemails', path='./data/', unzip=True)
print('✓ Phishing emails dataset downloaded!')
"

# Download SMS spam dataset
python3 -c "
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files('uciml/sms-spam-collection-dataset', path='./data/', unzip=True)
print('✓ SMS spam dataset downloaded!')
"
```

### 6. Test the Pipeline
```bash
# Run data quality tests
python test_data_quality.py

# Run complete pipeline test
python test_complete_pipeline.py
```

## Troubleshooting

### Kaggle API Issues
- If you get "Missing username in configuration": Check that kaggle.json has both "username" and "key" fields
- If you get authentication errors: Verify permissions are 600 on kaggle.json
- If you get outdated API warning: This is normal with kaggle==1.6.17, ignore it

### Package Detection Issues
- The setup checker may fail to detect packages even if they're installed
- This is a known issue with the checker script
- Verify packages are installed with: `pip list | grep package_name`
- Test imports directly with Python as shown in step 4

### Virtual Environment Issues
- Always activate the venv before running scripts: `source .venv/bin/activate`
- Check which Python is being used: `which python`
- It should point to `.venv/bin/python`

## Notes
- The kaggle package version is locked to 1.6.17 for compatibility
- Newer versions (2.x) use different authentication methods
- The setup checker may show some warnings but the environment should work
- Direct Python import tests are more reliable than the checker script