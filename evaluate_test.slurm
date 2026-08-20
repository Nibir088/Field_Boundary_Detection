#!/bin/bash
#SBATCH --job-name=ftw-test
#SBATCH --account=nssac_students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=ftw-test-%j.out
#SBATCH --error=ftw-test-%j.err
#SBATCH --export=ALL

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
DATA_DIR="/sfs/weka/scratch/${USER}/ftw_data/ftw"
MODEL="/sfs/weka/scratch/${USER}/ftw_models/FTW_PRUE_EFNET_B5.ckpt"
RESULTS_DIR="/sfs/weka/scratch/${USER}/ftw_results/job_${SLURM_JOB_ID}"

module purge
module load miniforge/24.11.3-py3.12

cd "${REPO_DIR}"
source .Field_Boundary/bin/activate

mkdir -p "${RESULTS_DIR}"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Dataset directory not found: ${DATA_DIR}" >&2
    exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
    echo "Model checkpoint not found: ${MODEL}" >&2
    exit 1
fi

echo "Evaluation started: $(date)"
echo "Compute node: $(hostname)"
echo "Dataset: ${DATA_DIR}"
echo "Model: ${MODEL}"
echo "Results: ${RESULTS_DIR}"

nvidia-smi

# Evaluate Belgium's 198 predefined test samples.
uv run ftw model test \
    --model="${MODEL}" \
    --dir="${DATA_DIR}" \
    --countries=belgium \
    --gpu=0 \
    --model_predicts_3_classes \
    --num_workers="${SLURM_CPUS_PER_TASK}" \
    --out="${RESULTS_DIR}/belgium_metrics.csv"

# Evaluate Rwanda's 7 predefined test samples.
uv run ftw model test \
    --model="${MODEL}" \
    --dir="${DATA_DIR}" \
    --countries=rwanda \
    --gpu=0 \
    --model_predicts_3_classes \
    --num_workers="${SLURM_CPUS_PER_TASK}" \
    --out="${RESULTS_DIR}/rwanda_metrics.csv"

# Evaluate the combined 205-sample test set.
uv run ftw model test \
    --model="${MODEL}" \
    --dir="${DATA_DIR}" \
    --countries=belgium \
    --countries=rwanda \
    --gpu=0 \
    --model_predicts_3_classes \
    --num_workers="${SLURM_CPUS_PER_TASK}" \
    --out="${RESULTS_DIR}/combined_metrics.csv"

echo "Evaluation completed: $(date)"
echo
echo "Belgium metrics"
cat "${RESULTS_DIR}/belgium_metrics.csv"
echo
echo "Rwanda metrics"
cat "${RESULTS_DIR}/rwanda_metrics.csv"
echo
echo "Combined metrics"
cat "${RESULTS_DIR}/combined_metrics.csv"
