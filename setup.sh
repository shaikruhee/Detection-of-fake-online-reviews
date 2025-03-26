#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y python3-pip python3-venv build-essential python3-dev

# Install Python packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

