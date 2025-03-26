#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y python3-pip python3-venv build-essential python3-dev

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python packages
pip install -r requirements.txt
pip install joblib

# Run the Streamlit app
streamlit run app.py

