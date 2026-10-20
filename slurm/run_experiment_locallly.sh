#!/bin/bash
#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_DIR="logs/log_${TIMESTAMP}"

mkdir -p "$LOG_DIR"

experiments=(
    # c01
    # c02_ia
    # ca
    # ca_ia
    # ca_ib
    # ca_ic
    ca_ia_ic
    ca_cb_ia_ic
)

for EXP in "${experiments[@]}"; do
    echo "Running $EXP"

    python main.py +experiment=$EXP \
        > "${LOG_DIR}/${EXP}.out" \
        2> "${LOG_DIR}/${EXP}.err"

done

echo "Logs saved in ${LOG_DIR}"


