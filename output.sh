#!/bin/bash

# ============================================================
#  Step 0: 기본 설정
# ============================================================
dates=(
# forecast2
#    "2021-12-20" "2021-12-13" "2021-12-06" "2021-11-29" 
#    "2021-11-22" "2021-11-15" "2021-11-08" "2021-11-01"
#    "2021-10-25" "2021-10-18" "2021-10-11" "2021-10-04"
#    "2021-09-27" "2021-09-20" "2021-09-13" "2021-09-06"
   
#    "2021-08-30" "2021-08-23" "2021-08-16" "2021-08-09" 
#    "2021-08-02" "2021-07-26" "2021-07-19" "2021-07-12" 
   
<<<<<<< HEAD
# forecast2 retry
    # "2021-04-19" "2021-11-29" "2021-11-22" "2021-11-01"
    # "2021-09-27" "2021-09-20" "2021-08-09" "2021-07-26" 
    # "2021-07-19" "2021-07-12"  
    # "2021-09-20" "2021-07-19"
    # "2021-12-27"

=======
>>>>>>> a06c6b2018e86056c45458289a637d9b33476f4d
# forecast1
#    "2021-07-05" "2021-06-28" "2021-06-21" "2021-06-14" 
#    "2021-06-07" "2021-05-31" "2021-05-24" "2021-05-17"
#    "2021-05-10" "2021-05-03" "2021-04-26" "2021-04-19" 

#    "2021-04-12" "2021-04-05" "2021-03-29" "2021-03-22" 
#    "2021-03-15" "2021-03-08" "2021-03-01"
#    "2021-02-22" "2021-02-15" "2021-02-08" "2021-02-01"
<<<<<<< HEAD
     "2021-01-11" "2021-01-04" "2021-01-25" "2021-01-18" "2021-02-08" "2021-02-15"
)


=======
    "2021-01-25" "2021-01-18" # "2021-01-11" "2021-01-04"

# forecast1 retry
#     "2021-11-01" "2021-03-01" "2021-02-01"
)




>>>>>>> a06c6b2018e86056c45458289a637d9b33476f4d
MODEL="1.0_2019"
EVAL_STEPS="40"
ENS_NUM="10"

# 입력 및 출력 경로 설정
INPUT_DIR="/geodata2/Gencast/input/2021"
OUTPUT_DIR="/geodata2/Gencast/output/2021"
PYTHON_SCRIPT="/home/hiskim1/gencast/GenCast_run.py"

<<<<<<< HEAD
MAX_RETRIES=3  # 최대 재시도 횟수

# ============================================================
#  Step 1: 날짜별 실행 및 실패 시 재시도
# ============================================================
for date in "${dates[@]}"; do
    echo "$date started" >&2
    
    INPUT_FILE="$INPUT_DIR/$date.nc"
    OUTPUT_FILE="$OUTPUT_DIR/$date.zarr"

=======
# ============================================================
#  Step 1: 2025년 1월 2일부터 16일까지 반복 실행
# ============================================================
for date in "${dates[@]}";
do
    echo "$date started" >&2
    # 날짜 포맷 맞추기 (01, 02 등)
    DAY_FMT=$(printf "%02d" $day)
    
    INPUT_FILE="$INPUT_DIR/$date.nc"
    OUTPUT_FILE="$OUTPUT_DIR/$date.zarr"
    
>>>>>>> a06c6b2018e86056c45458289a637d9b33476f4d
    echo "-----------------------------------------------------------"
    echo "Running GenCast for $date..."
    echo "Input:  $INPUT_FILE"
    echo "Output: $OUTPUT_FILE"
    echo "-----------------------------------------------------------"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output file exists. Removing $OUTPUT_FILE..."
        rm -f "$OUTPUT_FILE"
    fi

<<<<<<< HEAD
    attempt=0
    while [ $attempt -lt $MAX_RETRIES ]; do
        python "$PYTHON_SCRIPT" \
            --model "$MODEL" \
            --eval_steps "$EVAL_STEPS" \
            --ens_num "$ENS_NUM" \
            --input "$INPUT_FILE" \
            --output "$OUTPUT_FILE"

        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "$date completed successfully." >&2
            break
        else
            echo "Error: Python script failed (attempt $((attempt + 1))/$MAX_RETRIES)." >&2
            ((attempt++))
            sleep 5  # 재시도 전 잠시 대기
        fi
    done

    if [ $attempt -eq $MAX_RETRIES ]; then
        echo "Failed to process $date after $MAX_RETRIES attempts. Skipping..." >&2
    fi
=======
    python "$PYTHON_SCRIPT" \
        --model "$MODEL" \
        --eval_steps "$EVAL_STEPS" \
        --ens_num "$ENS_NUM" \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE"
>>>>>>> a06c6b2018e86056c45458289a637d9b33476f4d

    echo "$date ended" >&2
    echo "=========================================================" >&2
done

echo "All GenCast runs completed!"