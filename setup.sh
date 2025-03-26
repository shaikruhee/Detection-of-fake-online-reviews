#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y python3-pip python3-venv build-essential python3-dev

# Upgrade pip
pip install --upgrade pip

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python packages with no cache\pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir joblib

# Run the Streamlit app
streamlit run app.py
