#!/bin/bash
#SBATCH --job-name=ftw-evaluation
#SBATCH --account=nssac_students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=ftw-evaluation-%j.out
#SBATCH --error=ftw-evaluation-%j.err
#SBATCH --export=ALL

set -euo pipefail

# ---------------------------------------------------------------------------
# EDIT THIS BLOCK
# Choose: delineate_anything or instance_boundary
EVALUATOR="delineate_anything"

DATA_DIR="/sfs/weka/scratch/${USER}/ftw_data/ftw"
RESULTS_ROOT="/sfs/weka/scratch/${USER}/ftw_results"
# Optional existing directory, useful when rerunning metrics without replacing PDFs.
RESULTS_DIR_OVERRIDE=""

# Delineate Anything settings (used only when EVALUATOR=delineate_anything).
DELANY_MODEL="DelineateAnythingV2"
DELANY_MODEL_PATH="/sfs/weka/scratch/${USER}/ftw_models/DelineateAnythingV2.pt"
DELANY_WINDOW="window_b"
DELANY_BATCH_SIZE=16

# Semantic instance/boundary settings (used only when EVALUATOR=instance_boundary).
SEMANTIC_MODEL="/sfs/weka/scratch/${USER}/ftw_models/FTW_PRUE_EFNET_B5.ckpt"
SEMANTIC_BATCH_SIZE=32
SEMANTIC_NUM_WORKERS=2

# Shared metric settings.
METRES_PER_PIXEL=10
GEOMETRY_EPSILON_PX=1
WRITE_SAMPLE_PDFS=true
WRITE_SUMMARY_PLOTS=true
# ---------------------------------------------------------------------------

REPO_DIR="${SLURM_SUBMIT_DIR}"
if [[ -n "${RESULTS_DIR_OVERRIDE}" ]]; then
    RESULTS_DIR="${RESULTS_DIR_OVERRIDE}"
else
    RESULTS_DIR="${RESULTS_ROOT}/${EVALUATOR}_job_${SLURM_JOB_ID}"
fi

module purge
module load miniforge/24.11.3-py3.12

cd "${REPO_DIR}"
source .Field_Boundary/bin/activate

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Dataset directory not found: ${DATA_DIR}" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "Evaluation started: $(date)"
echo "Compute node: $(hostname)"
echo "Evaluator: ${EVALUATOR}"
echo "Dataset: ${DATA_DIR}"
echo "Results: ${RESULTS_DIR}"
nvidia-smi

OPTIONAL_ARGS=()
if [[ "${WRITE_SAMPLE_PDFS}" != "true" ]]; then
    OPTIONAL_ARGS+=(--skip-sample-pdfs)
fi
if [[ "${WRITE_SUMMARY_PLOTS}" != "true" ]]; then
    OPTIONAL_ARGS+=(--skip-plots)
fi

case "${EVALUATOR}" in
    delineate_anything)
        if [[ ! -f "${DELANY_MODEL_PATH}" ]]; then
            echo "Delineate Anything checkpoint not found: ${DELANY_MODEL_PATH}" >&2
            exit 1
        fi
        python -m evaluation.delineate_anything \
            --model "${DELANY_MODEL}" \
            --model-path "${DELANY_MODEL_PATH}" \
            --data-dir "${DATA_DIR}" \
            --output-dir "${RESULTS_DIR}" \
            --window "${DELANY_WINDOW}" \
            --gpu 0 \
            --batch-size "${DELANY_BATCH_SIZE}" \
            --metres-per-pixel "${METRES_PER_PIXEL}" \
            --geometry-epsilon-px "${GEOMETRY_EPSILON_PX}" \
            "${OPTIONAL_ARGS[@]}"
        ;;
    instance_boundary)
        if [[ ! -f "${SEMANTIC_MODEL}" ]]; then
            echo "Semantic checkpoint not found: ${SEMANTIC_MODEL}" >&2
            exit 1
        fi
        python -m evaluation.instance_boundary \
            --model "${SEMANTIC_MODEL}" \
            --data-dir "${DATA_DIR}" \
            --output-dir "${RESULTS_DIR}" \
            --gpu 0 \
            --batch-size "${SEMANTIC_BATCH_SIZE}" \
            --num-workers "${SEMANTIC_NUM_WORKERS}" \
            --metres-per-pixel "${METRES_PER_PIXEL}" \
            --geometry-epsilon-px "${GEOMETRY_EPSILON_PX}" \
            "${OPTIONAL_ARGS[@]}"
        ;;
    *)
        echo "Unknown EVALUATOR=${EVALUATOR}" >&2
        echo "Choose delineate_anything or instance_boundary." >&2
        exit 1
        ;;
esac

echo "Evaluation completed: $(date)"
echo "CSV files and figures: ${RESULTS_DIR}"
du -sh "${RESULTS_DIR}"
