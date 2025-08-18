# CUAD Local Data Setup and Usage

## Setup

Since HuggingFace datasets no longer supports loading scripts, we've created a local data loading solution.

### 1. Download CUAD Data

First, run the setup script to download CUAD data to a local directory:

```bash
python setup_cuad_data.py
```

This will:
- Create a `cuad-data/` directory
- Download the CUAD dataset files (train and test JSON files)
- Download the original dataset script for reference

### 2. Updated Scripts

The following scripts have been updated to use local data:

- **cuad-demo.py** - Updated to load directly from local JSON files
- **cuad-max-quality-at-cost.py** - Updated to use `cuad_data_loader.py`

### 3. Running the Scripts

#### Basic CUAD Demo
```bash
# Make sure OPENAI_API_KEY is set in .env or environment
source ../.env && export OPENAI_API_KEY

# Run from abacus-research directory
seed=0
exp_name="cuad-final-mab-k6-j4-budget50-seed${seed}"
python cuad-demo.py --k 6 --j 4 --sample-budget 50 --seed $seed --exp-name $exp_name --gpt4-mini-only
```

#### Max Quality at Cost
```bash
python cuad-max-quality-at-cost.py --constrained --gpt4-mini-only
```