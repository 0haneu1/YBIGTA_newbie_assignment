#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
if ! command -v conda &>/dev/null; then
    echo "Miniconda가 존재하지 않습니다. 설치를 시작합니다..."
    # Linux ARM64용 (aarch64) Miniconda 설치 스크립트 다운로드
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
else
    echo "Conda가 이미 설치되어 있습니다."
    eval "$(conda shell.bash hook)"
fi

# Conda 환경 생성 및 활성화
if conda info --envs | grep -q "myenv"; then
    echo "이미 myenv  환경이 존재합니다. 활성화합니다."
else
    echo "myenv 환경이 존재하지 않습니다. 생성합니다."
    conda create -y -n myenv python=3.9
fi
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "가상환경 활성화: 성공"
else
    echo "가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
pip install mypy
echo "패키지 설치 완료. 다음 단계로 진행합니다..."

# Submission 폴더 파일 실행
cd submission || { echo "submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    base=$(basename "$file" .py)
    input_file="../input/${base}_input"
    output_file="../output/${base}_output"
    if [ -f "$input_file" ]; then
        echo "실행 중: $file (입력: $input_file → 출력: $output_file)"
        python "$file" < "$input_file" > "$output_file"
    else
        echo "실행 중: $file (입력 파일 없음 → 출력: $output_file)"
        python "$file" > "$output_file"
    fi
done

# mypy 테스트 실행
mypy *.py

# 가상환경 비활성화
conda deactivate

