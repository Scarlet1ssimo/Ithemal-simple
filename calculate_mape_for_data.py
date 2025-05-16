import pandas as pd
import torch
import io

# MAPE loss function (consistent with evaluate.py)
def mape_loss(predictions, targets, epsilon=1e-8):
    """Calculates Mean Absolute Percentage Error, handling potential division by zero."""
    absolute_percentage_error = torch.abs(
        (predictions - targets) / (targets + epsilon))
    return torch.mean(absolute_percentage_error)

def main():
    # Read the CSV data from the file
    try:
        df = pd.read_csv("evaluation_results_skl.csv")
    except FileNotFoundError:
        print("Error: 'evaluation_results_skl.csv' not found. Make sure the file is in the same directory as the script.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Ensure the relevant columns are numeric
    df['measured throughput'] = pd.to_numeric(df['measured throughput'])
    df['llvm-mca'] = pd.to_numeric(df['llvm-mca'])
    df['model'] = pd.to_numeric(df['model'])

    # Convert data to PyTorch tensors
    actual_throughput = torch.tensor(df['measured throughput'].values, dtype=torch.float32)
    llvm_mca_predictions = torch.tensor(df['llvm-mca'].values, dtype=torch.float32)
    model_predictions = torch.tensor(df['model'].values, dtype=torch.float32)

    # Calculate MAPE for llvm-mca vs actual
    mape_llvm_mca = mape_loss(llvm_mca_predictions, actual_throughput)

    # Calculate MAPE for model vs actual
    mape_model = mape_loss(model_predictions, actual_throughput)

    print(f"Data loaded from 'evaluation_results_skl.csv':")
    print(df)
    print("\n--- MAPE Results ---")
    print(f"MAPE (llvm-mca vs measured throughput): {mape_llvm_mca.item():.4f}")
    print(f"MAPE (model vs measured throughput):      {mape_model.item():.4f}")

if __name__ == '__main__':
    main()
