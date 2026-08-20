#!/bin/bash
#SBATCH --job-name=ftw-download
#SBATCH --account=nssac_students
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=ftw-download-%j.out
#SBATCH --error=ftw-download-%j.err
#SBATCH --export=ALL

set -euo pipefail

REPO_DIR="/scratch/${USER}/Field_Boundary_Detection"
DATA_DIR="/scratch/${USER}/ftw_data"

module purge
module load miniforge/24.11.3-py3.12

cd "${REPO_DIR}"
source .Field_Boundary/bin/activate

mkdir -p "${DATA_DIR}"

echo "Download started: $(date)"
echo "Compute node: $(hostname)"
echo "Destination: ${DATA_DIR}"

uv run ftw data download \
    --countries=all \
    --out="${DATA_DIR}"

echo "Download completed: $(date)"
du -sh "${DATA_DIR}"
