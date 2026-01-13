#!/bin/bash
# =============================================================================
# Master Launcher Script for ASVspoof 5 Domain-Invariant CM Pipeline
# Submits all jobs with proper dependency management to SLURM queue
#
# Usage:
#   ./scripts/jobs/submit_all.sh [--dry-run] [--skip-baselines] [--skip-held-out]
# =============================================================================

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
DRY_RUN=false
SKIP_BASELINES=false
SKIP_HELD_OUT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-baselines)
            SKIP_BASELINES=true
            shift
            ;;
        --skip-held-out)
            SKIP_HELD_OUT=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN MODE - No jobs will be submitted${NC}"
    echo ""
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ASVspoof 5 Domain-Invariant CM       ${NC}"
echo -e "${BLUE}  Pipeline Job Submission              ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Job scripts
SETUP_SCRIPT="scripts/jobs/setup_environment.job"
TRAIN_SCRIPTS=(
    "scripts/jobs/train_wavlm_erm.job"
    "scripts/jobs/train_wavlm_dann.job"
    "scripts/jobs/train_w2v2_erm.job"
    "scripts/jobs/train_w2v2_dann.job"
)
EVAL_SCRIPT="scripts/jobs/evaluate_models.job"
ANALYSIS_SCRIPT="scripts/jobs/run_analysis.job"
BASELINES_SCRIPT="scripts/jobs/run_baselines.job"
HELD_OUT_SCRIPT="scripts/jobs/run_held_out.job"

# Track job IDs
declare -a ALL_JOB_IDS
declare -a TRAIN_JOB_IDS

# Function to extract job info
get_job_info() {
    local script=$1
    if [ -f "$script" ]; then
        local job_name=$(grep -m1 "#SBATCH --job-name=" "$script" | cut -d'=' -f2)
        local partition=$(grep -m1 "#SBATCH --partition=" "$script" | cut -d'=' -f2)
        local time=$(grep -m1 "#SBATCH --time=" "$script" | cut -d'=' -f2)
        echo "$job_name ($partition, $time)"
    else
        echo "${RED}NOT FOUND${NC}"
    fi
}

# Show job summary
echo "Jobs to submit:"
echo "---------------"
echo ""
echo -e "  ${GREEN}Phase 1: Setup${NC}"
echo "    - $(get_job_info $SETUP_SCRIPT)"
echo ""
echo -e "  ${GREEN}Phase 2: Training (parallel)${NC}"
for script in "${TRAIN_SCRIPTS[@]}"; do
    echo "    - $(get_job_info $script)"
done
echo ""
if [ "$SKIP_BASELINES" = false ]; then
    echo -e "  ${GREEN}Phase 2b: Baselines (parallel with training)${NC}"
    echo "    - $(get_job_info $BASELINES_SCRIPT)"
    echo ""
fi
echo -e "  ${GREEN}Phase 3: Evaluation (after training)${NC}"
echo "    - $(get_job_info $EVAL_SCRIPT)"
echo ""
echo -e "  ${GREEN}Phase 4: Analysis (after ERM+DANN training)${NC}"
echo "    - $(get_job_info $ANALYSIS_SCRIPT)"
echo ""
if [ "$SKIP_HELD_OUT" = false ]; then
    echo -e "  ${GREEN}Phase 5: Held-Out Codec Experiment${NC}"
    echo "    - $(get_job_info $HELD_OUT_SCRIPT)"
    echo ""
fi

# Confirmation
if [ "$DRY_RUN" = false ]; then
    read -p "Submit all jobs? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

echo "Submitting jobs..."
echo "==================="
echo ""

# Phase 1: Setup
echo -e "${BLUE}Phase 1: Setup${NC}"
SETUP_JOB_ID=""
if [ -f "$SETUP_SCRIPT" ]; then
    job_name=$(grep -m1 "#SBATCH --job-name=" "$SETUP_SCRIPT" | cut -d'=' -f2)
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC}"
        SETUP_JOB_ID="DRYRUN_SETUP"
    else
        output=$(sbatch "$SETUP_SCRIPT" 2>&1)
        if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
            SETUP_JOB_ID="${BASH_REMATCH[1]}"
            ALL_JOB_IDS+=("$SETUP_JOB_ID")
            echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $SETUP_JOB_ID)"
        else
            echo -e "  ${RED}Failed to submit: $output${NC}"
            exit 1
        fi
    fi
else
    echo -e "  ${RED}Setup script not found: $SETUP_SCRIPT${NC}"
    exit 1
fi
echo ""

# Phase 2: Training (depends on setup)
echo -e "${BLUE}Phase 2: Training${NC}"
for script in "${TRAIN_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        job_name=$(grep -m1 "#SBATCH --job-name=" "$script" | cut -d'=' -f2)
        
        if [ "$DRY_RUN" = true ]; then
            echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after setup)"
        else
            output=$(sbatch --dependency=afterok:$SETUP_JOB_ID "$script" 2>&1)
            if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
                job_id="${BASH_REMATCH[1]}"
                ALL_JOB_IDS+=("$job_id")
                TRAIN_JOB_IDS+=("$job_id")
                echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id, depends on: $SETUP_JOB_ID)"
            else
                echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
            fi
        fi
    fi
done
echo ""

# Phase 2b: Baselines (depends on setup, parallel with training)
if [ "$SKIP_BASELINES" = false ]; then
    echo -e "${BLUE}Phase 2b: Baselines${NC}"
    if [ -f "$BASELINES_SCRIPT" ]; then
        job_name=$(grep -m1 "#SBATCH --job-name=" "$BASELINES_SCRIPT" | cut -d'=' -f2)
        
        if [ "$DRY_RUN" = true ]; then
            echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after setup)"
        else
            output=$(sbatch --dependency=afterok:$SETUP_JOB_ID "$BASELINES_SCRIPT" 2>&1)
            if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
                job_id="${BASH_REMATCH[1]}"
                ALL_JOB_IDS+=("$job_id")
                echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id, depends on: $SETUP_JOB_ID)"
            else
                echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
            fi
        fi
    fi
    echo ""
fi

# Phase 3: Evaluation (depends on all training jobs)
echo -e "${BLUE}Phase 3: Evaluation${NC}"
if [ -f "$EVAL_SCRIPT" ]; then
    job_name=$(grep -m1 "#SBATCH --job-name=" "$EVAL_SCRIPT" | cut -d'=' -f2)
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after training)"
    else
        # Build dependency string for all training jobs
        TRAIN_DEPS=$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")
        output=$(sbatch --dependency=afterok:$TRAIN_DEPS "$EVAL_SCRIPT" 2>&1)
        if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
            job_id="${BASH_REMATCH[1]}"
            ALL_JOB_IDS+=("$job_id")
            echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id, depends on: $TRAIN_DEPS)"
        else
            echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
        fi
    fi
fi
echo ""

# Phase 4: Analysis (depends on training - specifically ERM and DANN pairs)
echo -e "${BLUE}Phase 4: Analysis${NC}"
if [ -f "$ANALYSIS_SCRIPT" ]; then
    job_name=$(grep -m1 "#SBATCH --job-name=" "$ANALYSIS_SCRIPT" | cut -d'=' -f2)
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after training)"
    else
        TRAIN_DEPS=$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")
        output=$(sbatch --dependency=afterok:$TRAIN_DEPS "$ANALYSIS_SCRIPT" 2>&1)
        if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
            job_id="${BASH_REMATCH[1]}"
            ALL_JOB_IDS+=("$job_id")
            echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id, depends on: $TRAIN_DEPS)"
        else
            echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
        fi
    fi
fi
echo ""

# Phase 5: Held-Out Codec Experiment (depends on setup only - does its own training)
if [ "$SKIP_HELD_OUT" = false ]; then
    echo -e "${BLUE}Phase 5: Held-Out Codec Experiment${NC}"
    if [ -f "$HELD_OUT_SCRIPT" ]; then
        job_name=$(grep -m1 "#SBATCH --job-name=" "$HELD_OUT_SCRIPT" | cut -d'=' -f2)
        
        if [ "$DRY_RUN" = true ]; then
            echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after setup)"
        else
            output=$(sbatch --dependency=afterok:$SETUP_JOB_ID "$HELD_OUT_SCRIPT" 2>&1)
            if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
                job_id="${BASH_REMATCH[1]}"
                ALL_JOB_IDS+=("$job_id")
                echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id, depends on: $SETUP_JOB_ID)"
            else
                echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
            fi
        fi
    fi
    echo ""
fi

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Summary                               ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "Would submit ${GREEN}${#ALL_JOB_IDS[@]}${NC} jobs"
else
    echo -e "Submitted ${GREEN}${#ALL_JOB_IDS[@]}${NC} jobs"
    echo ""
    echo "Job IDs: ${ALL_JOB_IDS[*]}"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Cancel all with:"
    echo "  scancel ${ALL_JOB_IDS[*]}"
fi
echo ""
