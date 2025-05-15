import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os
import subprocess
import csv
from tqdm import tqdm
import re
import sys
import argparse # Added for command-line arguments
from concurrent.futures import ProcessPoolExecutor # Added for parallelization

# Assuming these are in the same directory or PYTHONPATH is set
# Adjust imports based on your project structure if these files are elsewhere
try:
    from bhive_dataloader import BHiveDataset, collate_fn, _TOKENIZER, _fake_intel
    from model import IthemalRNN
    import data_cost as dt
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure evaluate.py is in the correct directory or PYTHONPATH is set.")
    sys.exit(1)

# Configuration
TARGET = os.environ.get("ITHEMAL_TARGET", "skl")
MODEL_PATH = "my_models/ithemal_bhive_skl_best.pt"
BHIVE_THROUGHPUT_FILE_TEMPLATE = "bhive/benchmark/throughput/{}.csv"
VOCAB_MAP_PATH_TEMPLATE = "vocab_map_{}.pkl"
OUTPUT_CSV_FILE_TEMPLATE = "evaluation_results_{}.csv"
DISASM_PATH = "/home/scarlet/521/mp4/bhive/benchmark/disasm" # Path to disasm executable
LLVM_MCA_COMMAND = ["llvm-mca", "--iterations=1", "-mcpu=skylake"] # llvm-mca command and args

# Use MAPE loss as defined in train.py (moved to top level)
def mape_loss(predictions, targets, epsilon=1e-8):
    """Calculates Mean Absolute Percentage Error, handling potential division by zero."""
    absolute_percentage_error = torch.abs(
        (predictions - targets) / (targets + epsilon))
    return torch.mean(absolute_percentage_error) # Return mean for overall loss

# Worker function for ProcessPoolExecutor (moved to top level)
def process_mca_item_worker(item_data):
    hex_string = item_data['hex']
    true_cycles_original = item_data['true_cycles_original']
    # model_pred_original_val is part of item_data but not used for mca loss calculation directly in this worker

    mca_pred_cycles_raw_val = get_llvm_mca_throughput(hex_string) # I/O and CPU bound
    loss_mca_item_val = float('nan')

    if mca_pred_cycles_raw_val is not None:
        mca_value_for_loss = mca_pred_cycles_raw_val * 100.0
        # Perform loss calculation on CPU to avoid CUDA issues in worker processes
        cpu_device = torch.device('cpu')
        loss_mca_item_val = mape_loss( # Call top-level mape_loss
            torch.tensor([mca_value_for_loss], device=cpu_device),
            torch.tensor([true_cycles_original], device=cpu_device)
        ).item()

    return {
        'hex': hex_string,
        'true_cycles_original': true_cycles_original,
        'model_pred_original_val': item_data['model_pred_original_val'], # Pass through for CSV
        'mca_pred_cycles_raw_val': mca_pred_cycles_raw_val if mca_pred_cycles_raw_val is not None else float('nan'),
        'loss_mca_item': loss_mca_item_val
    }

# Helper to run llvm-mca
def get_llvm_mca_throughput(hex_string):
    if not os.path.exists(DISASM_PATH):
        print(f"Error: disasm executable not found at {DISASM_PATH}", file=sys.stderr)
        return None
    if not os.access(DISASM_PATH, os.X_OK):
        print(f"Error: disasm executable at {DISASM_PATH} is not executable.", file=sys.stderr)
        return None

    try:
        # Step 1: Run disasm
        disasm_process = subprocess.Popen(
            [DISASM_PATH, hex_string],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        disasm_stdout, disasm_stderr = disasm_process.communicate(timeout=30)

        if disasm_process.returncode != 0:
            print(f"Warning: disasm failed for {hex_string[:20]}... Return code: {disasm_process.returncode}. Error: {disasm_stderr.strip()}", file=sys.stderr)
            return None

        # Step 2: Run llvm-mca, piping disasm output to its stdin
        mca_process = subprocess.Popen(
            LLVM_MCA_COMMAND,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        mca_stdout, mca_stderr = mca_process.communicate(input=disasm_stdout, timeout=30)

        if mca_process.returncode != 0:
            print(f"Warning: llvm-mca failed for {hex_string[:20]}... Return code: {mca_process.returncode}. Error: {mca_stderr.strip()}", file=sys.stderr)
            return None

        # Step 3: Parse llvm-mca output (equivalent to grep RThroughput:)
        for line in mca_stdout.splitlines():
            match = re.search(r"Block RThroughput:\s*([0-9.]+)", line)
            if match:
                return float(match.group(1))*100
        
        print(f"Warning: Could not parse RThroughput for {hex_string[:20]}... MCA Output: {mca_stdout.strip()}", file=sys.stderr)
        return None

    except subprocess.TimeoutExpired:
        print(f"Timeout during llvm-mca pipeline for {hex_string[:20]}...", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error during llvm-mca pipeline for {hex_string[:20]}...: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate Ithemal model and compare with LLVM-MCA.")
    parser.add_argument("--test_first_n", type=int, default=0,
                        help="Number of initial samples to process for testing. Default is 0 (process all).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check ITHEMAL_HOME
    if 'ITHEMAL_HOME' not in os.environ:
        print("Error: ITHEMAL_HOME environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(os.environ['ITHEMAL_HOME']):
        print(f"Error: ITHEMAL_HOME ('{os.environ['ITHEMAL_HOME']}') is not a valid directory.", file=sys.stderr)
        sys.exit(1)
    
    # Check tokenizer
    if not os.path.exists(_TOKENIZER):
         print(f"Error: Tokenizer not found at {_TOKENIZER}. Check ITHEMAL_HOME setting and data_collection build.", file=sys.stderr)
         sys.exit(1)

    vocab_map_path = VOCAB_MAP_PATH_TEMPLATE.format(TARGET)
    if not os.path.exists(vocab_map_path):
        print(f"Error: Vocabulary map '{vocab_map_path}' not found. Please generate it first.", file=sys.stderr)
        sys.exit(1)

    try:
        token_idx_map = dt.load_token_idx_map(vocab_map_path)
        if not isinstance(token_idx_map, dt.DataInstructionEmbedding):
             print(f"Error: Loaded vocab_map from {vocab_map_path} is not a DataInstructionEmbedding instance.", file=sys.stderr)
             sys.exit(1)
        vocab_size = len(token_idx_map.token_to_hot_idx)
    except Exception as e:
        print(f"Error loading or processing vocabulary map {vocab_map_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Load Model
    embedding_size = 256  # Standard Ithemal value, adjust if your model differs
    hidden_size = 256     # Standard Ithemal value, adjust if your model differs
    model = IthemalRNN(vocab_size=vocab_size, embedding_size=embedding_size, hidden_size=hidden_size)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model state_dict from {MODEL_PATH}: {e}", file=sys.stderr)
        print("Ensure model architecture in model.py matches the saved model.", file=sys.stderr)
        sys.exit(1)
    
    model.to(device)
    model.eval()

    # criterion will use the top-level mape_loss function
    criterion = mape_loss

    throughput_file = BHIVE_THROUGHPUT_FILE_TEMPLATE.format(TARGET)
    if not os.path.exists(throughput_file):
        print(f"Error: Throughput data file not found at {throughput_file}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(throughput_file, header=None, names=['hex', 'throughput_original'])
        df.dropna(inplace=True)
    except Exception as e:
        print(f"Error reading throughput file {throughput_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize lists for storing results and losses
    all_model_losses_normalized_scale = []
    intermediate_data_for_mca = [] # To store data between the two loops

    # This instance is used for its _datum_of_code_internal method.
    try:
        dataset_processor = BHiveDataset(
            throughput_file=throughput_file, 
            token_idx_map_ref=token_idx_map,
            deserialize=False, 
            dump=False
        )
    except SystemExit:
        print("Critical error during BHiveDataset initialization for processor.", file=sys.stderr)
        sys.exit(1)

    processed_rows_model = 0 # Counter for model processing loop

    # First loop: Model predictions
    print("Starting Model Prediction Phase...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0] if args.test_first_n == 0 else args.test_first_n, desc="Model Predicting"):
        if args.test_first_n > 0 and processed_rows_model >= args.test_first_n:
            print(f"\nProcessed the first {args.test_first_n} rows for model prediction as requested.")
            break

        hex_string = row['hex']
        true_cycles_original = float(row['throughput_original'])
        model_pred_original_val = float('nan') # Initialize for this iteration

        try:
            data_item = dataset_processor._datum_of_code_internal(hex_string, verbose=False)
            if data_item is None:
                intermediate_data_for_mca.append({
                    'hex': hex_string,
                    'true_cycles_original': true_cycles_original,
                    'model_pred_original_val': model_pred_original_val # nan
                })
                processed_rows_model += 1
                continue
            
            true_throughput_normalized = torch.tensor(true_cycles_original / 100.0, dtype=torch.float32)
            data_item.y = true_throughput_normalized # Model expects normalized target

            batch = collate_fn([data_item])
            if batch is None:
                intermediate_data_for_mca.append({
                    'hex': hex_string,
                    'true_cycles_original': true_cycles_original,
                    'model_pred_original_val': model_pred_original_val # nan
                })
                processed_rows_model += 1
                continue

            with torch.no_grad():
                model_output_normalized = model(batch) 
                if model_output_normalized is not None and model_output_normalized.numel() > 0 :
                    model_pred_original_val = model_output_normalized.item() * 100.0 # Convert to original scale
                    
                    loss_model_item = criterion(torch.tensor([model_pred_original_val], device=device), 
                                                torch.tensor([true_cycles_original], device=device))
                    all_model_losses_normalized_scale.append(loss_model_item.item()) # Store MAPE
                # else: model_pred_original_val remains nan
        except Exception as e:
            pass 

        intermediate_data_for_mca.append({
            'hex': hex_string,
            'true_cycles_original': true_cycles_original,
            'model_pred_original_val': model_pred_original_val
        })
        processed_rows_model += 1

    avg_model_loss = sum(all_model_losses_normalized_scale) / len(all_model_losses_normalized_scale) if all_model_losses_normalized_scale else float('nan')
    print(f"\nModel Prediction Phase Complete.")
    print(f"Average Model MAPE (prediction vs true, original scale): {avg_model_loss:.4f} (lower is better)")

    # Second loop: LLVM-MCA evaluation (Parallelized)
    all_mca_losses_prompt_specific = []
    results_for_csv = [] # Initialize for final CSV results

    print("\nStarting LLVM-MCA Evaluation Phase (Parallelized)...")

    num_workers = os.cpu_count()
    print(f"Using {num_workers} workers for LLVM-MCA evaluation.")

    if intermediate_data_for_mca: # Only run if there's data
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # executor.map processes items and returns an iterator
            # tqdm wraps this iterator to show progress
            # process_mca_item_worker is now a top-level function
            future_results = executor.map(process_mca_item_worker, intermediate_data_for_mca)
            
            for result in tqdm(future_results, total=len(intermediate_data_for_mca), desc="LLVM-MCA Evaluating (Parallel)"):
                if not pd.isna(result['loss_mca_item']):
                    all_mca_losses_prompt_specific.append(result['loss_mca_item'])
                
                results_for_csv.append({
                    'hex': result['hex'],
                    'true_throughput_original': result['true_cycles_original'],
                    'mca_pred_cycles_raw': result['mca_pred_cycles_raw_val'],
                    'model_pred_original': result['model_pred_original_val']
                })
    else:
        print("No data to process for LLVM-MCA evaluation.")

    avg_mca_loss = sum(all_mca_losses_prompt_specific) / len(all_mca_losses_prompt_specific) if all_mca_losses_prompt_specific else float('nan')
    print(f"\nLLVM-MCA Evaluation Phase Complete.")

    print(f"\nEvaluation Complete (Final Summary):")
    print(f"Average Model MAPE (prediction vs true, original scale): {avg_model_loss:.4f} (lower is better)")
    print(f"Average MCA MAPE (prediction vs true, original scale): {avg_mca_loss:.4f} (lower is better)")

    output_csv_file = OUTPUT_CSV_FILE_TEMPLATE.format(TARGET)
    with open(output_csv_file, 'w', newline='') as csvfile:
        fieldnames = ['hex', 'measured throughput', 'llvm-mca', 'model']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results_for_csv:
            writer.writerow({
                'hex': res['hex'],
                'measured throughput': '%.2f' % res['true_throughput_original'] if not pd.isna(res['true_throughput_original']) else 'NaN',
                'llvm-mca': '%.2f' % res['mca_pred_cycles_raw'] if not pd.isna(res['mca_pred_cycles_raw']) else 'NaN',
                'model': '%.2f' % res['model_pred_original'] if not pd.isna(res['model_pred_original']) else 'NaN'
            })
    print(f"Results saved to {output_csv_file}")

if __name__ == '__main__':
    main()
