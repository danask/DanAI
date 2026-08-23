#!/bin/bash

# 1. 스크립트 위치 기준으로 디렉터리 이동
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 2. myenv 폴더 존재 여부 확인 및 생성
if [ ! -d "myenv" ]; then
    echo "myenv가 존재하지 않아 새로 생성합니다."
    python3 -m venv myenv
fi

# 3. 가상환경 내부 파이썬 바이너리 경로 지정
VENV_PYTHON="$DIR/myenv/bin/python"

# 4. 패키지 설치 및 uvicorn 실행 (가상환경 전용 파이썬 사용)
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements.txt
"$VENV_PYTHON" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000