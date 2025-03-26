import subprocess
import sys

# Run the setup script
result = subprocess.run(['bash', 'setup.sh'], text=True)

if result.returncode != 0:
    sys.exit(f"Setup failed with code {result.returncode}")

# Launch the Streamlit app
subprocess.run(['streamlit', 'run', 'app.py'])
