#!/bin/bash

# FLEURS Spanish ASR Evaluation Script for Cohere ASR (transformers-based)

export PYTHONPATH="..":$PYTHONPATH
RUNDIR=`pwd`
PYTHON="/admin/home/eustache_lebihan/cohere-asr/.venv/bin/python"

# Configuration
MODEL_IDs=(
    "eustlb/converted-cohere-asr"
)
BATCH_SIZE=256
DEVICE_ID=0

DATASETS="nithinraok/asr-leaderboard-datasets"

for MODEL_ID in "${MODEL_IDs[@]}"; do
    echo ""
    echo "Processing Model: $MODEL_ID"
    echo "========================================================"

    if [[ -d "$MODEL_ID" ]]; then
        short_model_id="${MODEL_ID##*/}"
    else
        short_model_id="${MODEL_ID#*/}"
    fi

    RESULTS_DIR="${RUNDIR}/benchmarks/results-${short_model_id}/ml"

    echo ""
    echo "Running evaluation: fleurs_es"
    echo "   Model: $MODEL_ID"
    echo "   Device: $DEVICE_ID"
    echo "   Batch Size: $BATCH_SIZE"
    echo "   Time: $(date)"
    echo "----------------------------------------"

    $PYTHON run_eval_trfms.py \
        --model_id="$MODEL_ID" \
        --dataset_path="$DATASETS" \
        --dataset="fleurs_es" \
        --split="test" \
        --device="$DEVICE_ID" \
        --batch_size="$BATCH_SIZE" \
        --max_eval_samples=-1 \
        --language="es" \
        --basedir="$RESULTS_DIR"

    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully for fleurs_es"
    else
        echo "Evaluation failed for fleurs_es (exit code: $?)"
    fi

    echo "----------------------------------------"

    echo ""
    echo "========================================================"
    echo "Evaluating results for $MODEL_ID"
    echo "========================================================"

    cd ../normalizer
    $PYTHON -c "import eval_utils; eval_utils.score_results('${RESULTS_DIR}', '${MODEL_ID}')"
    cd $RUNDIR

done

echo ""
echo "========================================================"
echo "All evaluations completed!"
echo "========================================================"
