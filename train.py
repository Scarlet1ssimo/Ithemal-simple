import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys
import argparse
import time
import pickle  # Added for loading vocab map

# Import necessary components from other files
import data_cost as dt
# Use BHive specific dataloader and collate
from bhive_dataloader import BHiveDataset, collate_fn
from model import IthemalRNN
import utilities as ut  # Added for potential utility functions

# --- Configuration ---
TARGET = os.environ.get("ITHEMAL_TARGET", "skl")  # Get target from env var
VOCAB_MAP_FILE = f'vocab_map_{TARGET}.pkl'
THROUGHPUT_FILE = os.path.join(
    'bhive', 'benchmark', 'throughput', f'{TARGET}.csv')


def train(args):
    """Main training loop."""

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    print(f"Using Target Architecture (TARGET): {TARGET}")

    # --- Load Vocabulary Map ---
    print(f"Loading vocabulary map from: {VOCAB_MAP_FILE}")
    try:
        token_idx_map_ref = dt.load_token_idx_map(VOCAB_MAP_FILE)
        vocab_size = len(token_idx_map_ref.token_to_hot_idx)
        print(f"Vocabulary map loaded. Vocabulary size: {vocab_size}")
    except FileNotFoundError:
        print(f"Error: Vocabulary map file not found at {VOCAB_MAP_FILE}")
        print("Please ensure the map is generated first (e.g., by running bhive_dataloader.py with 'create_map').")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load vocabulary map: {e}")
        sys.exit(1)

    # --- Data Loading and Splitting ---
    print(f"Loading dataset from: {THROUGHPUT_FILE}")
    try:
        dataset = BHiveDataset(
            throughput_file=THROUGHPUT_FILE, token_idx_map_ref=token_idx_map_ref, deserialize=True)
    except SystemExit:  # Catch sys.exit called by BHiveDataset on file error
        print(f"Exiting due to dataset loading error.")
        sys.exit(1)
    except Exception as e:
        print(
            f"An unexpected error occurred during dataset initialization: {e}")
        sys.exit(1)

    if len(dataset) == 0:
        print(
            "Error: Dataset is empty. Check the throughput file and data loading process.")
        sys.exit(1)

    print(f"Dataset loaded. Total samples: {len(dataset)}")

    # Split dataset into training and validation sets
    val_split = args.validation_split
    if not (0 < val_split < 1):
        print("Error: Validation split must be between 0 and 1.")
        sys.exit(1)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    if train_size == 0 or val_size == 0:
        print(
            f"Error: Dataset size ({len(dataset)}) is too small to create both training and validation sets with split {val_split}.")
        sys.exit(1)

    print(f"Splitting dataset: Train={train_size}, Validation={val_size}")
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size])

    # Create DataLoaders
    # Note: collate_fn currently returns a list of items. The loop handles this.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)
    print("DataLoaders created.")

    # --- Model Initialization ---
    print("Initializing model...")
    model = IthemalRNN(vocab_size=vocab_size,
                       embedding_size=args.embedding_size,
                       hidden_size=args.hidden_size).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # --- Loss and Optimizer ---
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Using Loss: {criterion}")
    print(f"Using Optimizer: Adam (lr={args.lr})")

    # --- Training Loop ---
    print("Starting training...")
    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()  # Set model to training mode
        epoch_train_loss = 0.0
        processed_train_items = 0

        # --- Training Phase ---
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for batch_data in progress_bar:
            if batch_data is None:  # Skip empty batches from collate_fn
                continue

            batch_loss = 0.0
            valid_items_in_batch = 0
            optimizer.zero_grad()

            # Process each item in the batch individually (due to current collate_fn)
            for item in batch_data:
                if item is None:  # Should not happen if collate_fn filters, but check anyway
                    continue
                target = item.y.to(device)  # Move target tensor to device

                try:
                    # Model forward pass expects a single DataItem
                    # Move prediction to device
                    prediction = model(item).to(device)

                    loss = criterion(prediction, target)
                    batch_loss += loss
                    valid_items_in_batch += 1

                except Exception as e:
                    print(
                        f"\nError during training forward pass (Code ID {item.code_id if hasattr(item, 'code_id') else 'N/A'}): {e}")
                    # Optionally add more debugging info or skip the item
                    continue  # Skip this item

            # Backpropagate and update weights if any valid items were processed
            if valid_items_in_batch > 0:
                avg_batch_loss = batch_loss / valid_items_in_batch
                avg_batch_loss.backward()
                optimizer.step()

                epoch_train_loss += avg_batch_loss.item() * valid_items_in_batch
                processed_train_items += valid_items_in_batch
                progress_bar.set_postfix(
                    batch_loss=f'{avg_batch_loss.item():.4f}')

        # --- End of Training Epoch ---
        avg_epoch_train_loss = epoch_train_loss / \
            processed_train_items if processed_train_items > 0 else 0
        progress_bar.close()

        # --- Validation Phase ---
        model.eval()  # Set model to evaluation mode
        epoch_val_loss = 0.0
        processed_val_items = 0
        val_progress_bar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)

        with torch.no_grad():  # Disable gradient calculations for validation
            for batch_data in val_progress_bar:
                if batch_data is None:
                    continue

                for item in batch_data:
                    if item is None:
                        continue
                    target = item.y.to(device)

                    try:
                        prediction = model(item).to(device)
                        loss = criterion(prediction, target)
                        epoch_val_loss += loss.item()
                        processed_val_items += 1
                    except Exception as e:
                        print(
                            f"\nError during validation forward pass (Code ID {item.code_id if hasattr(item, 'code_id') else 'N/A'}): {e}")
                        continue  # Skip this item

        avg_epoch_val_loss = epoch_val_loss / \
            processed_val_items if processed_val_items > 0 else 0
        val_progress_bar.close()

        # --- Epoch Summary ---
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Avg Training Loss: {avg_epoch_train_loss:.4f}")
        print(f"  Avg Validation Loss: {avg_epoch_val_loss:.4f}")
        print(f"  Time Elapsed: {elapsed_time:.2f}s")
        print("-" * 50)

        # --- Save Model Checkpoint (Best Model based on Validation Loss) ---
        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            if args.save_path:
                save_file = os.path.join(
                    args.save_path, f"ithemal_bhive_{TARGET}_best.pt")
                os.makedirs(args.save_path, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_train_loss,  # Save training loss as well
                    'val_loss': avg_epoch_val_loss,
                    'vocab_size': vocab_size,
                    'embedding_size': args.embedding_size,
                    'hidden_size': args.hidden_size,
                    'target_arch': TARGET,
                    'args': args  # Save args used for training
                }, save_file)
                print(
                    f"New best model saved to {save_file} (Val Loss: {best_val_loss:.4f})")

        # Optional: Save checkpoint every N epochs regardless of performance
        if args.save_path and args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            save_file_epoch = os.path.join(
                args.save_path, f"ithemal_bhive_{TARGET}_epoch_{epoch+1}.pt")
            os.makedirs(args.save_path, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_train_loss,
                'val_loss': avg_epoch_val_loss,
                'vocab_size': vocab_size,
                'embedding_size': args.embedding_size,
                'hidden_size': args.hidden_size,
                'target_arch': TARGET,
                'args': args
            }, save_file_epoch)
            print(f"Epoch {epoch+1} checkpoint saved to {save_file_epoch}")

    print("Training finished.")
    end_time = time.time()
    print(f"Total Training Time: {(end_time - start_time):.2f}s")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")

    # --- Optional: Save Final Model (usually the best one is preferred) ---
    # You might want to remove this or save it with a different name
    # if args.save_path:
    #     save_file = os.path.join(args.save_path, f"ithemal_bhive_{TARGET}_final.pt")
    #     os.makedirs(args.save_path, exist_ok=True)
    #     torch.save({ ... }, save_file) # Populate with final state if needed
    #     print(f"Final model state saved to {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Train Ithemal model on BHive data for TARGET={TARGET}.")
    # Removed --throughput-file as it's derived from TARGET env var
    parser.add_argument('--epochs', type=int, default=20,  # Increased default epochs
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,  # Adjusted default batch size
                        help='Number of samples per batch (processed individually in the loop)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--embedding-size', type=int, default=256,
                        help='Size of token embeddings')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Size of LSTM hidden states')
    parser.add_argument('--num-workers', type=int, default=4,  # Adjusted default workers
                        help='Number of dataloader workers')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Fraction of data to use for validation (e.g., 0.2 for 20%)')
    # parser.add_argument('--log-interval', type=int, default=50, # Replaced by tqdm progress bar
    #                     help='How many batches to wait before logging training status')
    parser.add_argument('--save-path', type=str, default='trained_models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save-interval', type=int, default=0,  # Default 0 means only save best model
                        help='Save a checkpoint every N epochs (0 to disable, only best model saved)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training even if available')

    # Add tqdm for progress bars
    try:
        from tqdm import tqdm
    except ImportError:
        print("Warning: tqdm not found. Install it (`pip install tqdm`) for progress bars.")
        # Define a dummy tqdm if not installed

        def tqdm(iterable, *args, **kwargs):
            return iterable

    args = parser.parse_args()

    # Validate derived throughput file path (optional but good practice)
    if not os.path.exists(THROUGHPUT_FILE):
        print(f"Error: Derived throughput file not found at {THROUGHPUT_FILE}")
        print(
            f"Ensure the TARGET environment variable ('{TARGET}') is correct and the file exists.")
        sys.exit(1)

    train(args)
