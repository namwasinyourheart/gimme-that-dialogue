#!/bin/bash

# Define the list of experiment config files
EXP_CONFIGS=(
    "configs/llama-3.2-3b-instruct__samsum.yaml"
    # "configs/bart-large-cnn__samsum.yaml"
    # "configs/t5-small__samsum.yaml"
    # "configs/flan-t5-base__samsum.yaml"
    # "configs/llama-3.2-1b-instruct__samsum.yaml"
    # "configs/qwen2.5-1.5b-instruct__samsum.yaml"
)

# Directory to store results
RESULTS_DIR="exps"

# Function to run an experiment
run_experiment() {
    CONFIG_PATH=$1
    EXP_NAME=$(basename "$CONFIG_PATH" .yaml)
    EXP_DIR="$RESULTS_DIR/$EXP_NAME"

    echo "🚀 Running experiment: $EXP_NAME"
    
    # Ensure the experiment directory exists
    mkdir -p "$EXP_DIR"

    # Step 1: Prepare data
    echo "🔹 Preparing data for $EXP_NAME..."
    python prepare_data.py --config_path "$CONFIG_PATH" prepare_data.dataset.is_prepared=False prepare_data.dataset.subset_ratio=1 exp_manager.print_cfg=False

    # Step 2: Run evaluation
    echo "🔹 Evaluating $EXP_NAME..."
    python eval.py --config_path "$CONFIG_PATH" prepare_data.dataset.is_prepared=True exp_manager.print_cfg=False eval.break_step=-1

    echo "✅ Experiment '$EXP_NAME' completed!"
    echo "----------------------------------------"
}

# Loop through all experiments and execute them
for CONFIG in "${EXP_CONFIGS[@]}"; do
    run_experiment "$CONFIG"
    # rm -rf ~/.cache/huggingface/hub
done

echo "🎯 All experiments finished successfully!"

echo "📊 Summarizing results..."
python -c 'from summarize_results import summarize_results; summarize_results()'
