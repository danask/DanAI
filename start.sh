#!/bin/bash

# 1. Move to the directory this script lives in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 2. Create the myenv virtual environment if it doesn't exist yet
if [ ! -d "myenv" ]; then
    echo "myenv does not exist, creating a new one."
    python3 -m venv myenv
fi

# 3. Path to the Python binary inside the virtual environment
VENV_PYTHON="$DIR/myenv/bin/python"

# 4. Install packages and run uvicorn (using the venv's own Python)
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements.txt
"$VENV_PYTHON" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
