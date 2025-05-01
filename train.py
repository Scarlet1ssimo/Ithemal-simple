import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys
import argparse
import time
import pickle
from tqdm import tqdm  # Ensure tqdm is imported

# Import necessary components from other files
import data_cost as dt
# Use BHive specific dataloader and collate
from bhive_dataloader import BHiveDataset, collate_fn  # collate_fn is now updated
from model import IthemalRNN
import utilities as ut

# --- Custom Loss Function ---


def mape_loss(predictions, targets, epsilon=1e-8):
    """Calculates Mean Absolute Percentage Error, handling potential division by zero."""
    # Ensure targets are not zero or very close to zero
    # Add epsilon to the denominator to avoid division by zero
    # Use torch.abs for absolute difference
    absolute_percentage_error = torch.abs(
        (predictions - targets) / (targets + epsilon))
    return torch.mean(absolute_percentage_error)


# --- Configuration ---
TARGET = os.environ.get("ITHEMAL_TARGET", "skl")
VOCAB_MAP_FILE = f'vocab_map_{TARGET}.pkl'
THROUGHPUT_FILE = os.path.join(
    'bhive', 'benchmark', 'throughput', f'{TARGET}.csv')
# Define padding_idx based on your vocab map or assume 0 if not explicitly defined
PADDING_IDX = 0


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
        # Ensure the vocab map has the padding token if needed, or handle it here
        if '<pad>' not in token_idx_map_ref.token_to_hot_idx:
            print(
                "Warning: '<pad>' token not found in vocab map. Assuming index 0 is padding.")
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
            throughput_file=THROUGHPUT_FILE, token_idx_map_ref=token_idx_map_ref, deserialize=args.deserialize)
    except SystemExit:
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

    # Create DataLoaders with the updated collate_fn
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    print("DataLoaders created.")

    # --- Model Initialization ---
    print("Initializing model...")
    model = IthemalRNN(vocab_size=vocab_size,
                       embedding_size=args.embedding_size,
                       hidden_size=args.hidden_size,
                       padding_idx=PADDING_IDX).to(device)

    # --- Loss and Optimizer ---
    # Replace MSELoss with the custom MAPE loss function
    criterion = mape_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Updated print statement
    print(f"Using Loss: MAPE (|pred-actual|/actual)")
    print(f"Using Optimizer: Adam (lr={args.lr})")

    start_epoch = 0
    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"Loading model checkpoint from: {args.load_model}")
            # Explicitly set weights_only=False to load the full checkpoint, including argparse.Namespace
            # Ensure you trust the source of this checkpoint file.
            checkpoint = torch.load(
                args.load_model, map_location=device, weights_only=False)

            # --- Parameter Verification (Optional but Recommended) ---
            loaded_args = checkpoint.get('args')
            loaded_vocab_size = checkpoint.get('vocab_size')
            loaded_embedding_size = checkpoint.get('embedding_size')
            loaded_hidden_size = checkpoint.get('hidden_size')
            loaded_padding_idx = checkpoint.get(
                'padding_idx', 0)  # Default to 0 if not saved

            mismatched_params = []
            if loaded_vocab_size is not None and loaded_vocab_size != vocab_size:
                mismatched_params.append(
                    f"Vocab Size (loaded {loaded_vocab_size}, current {vocab_size})")
            if loaded_embedding_size is not None and loaded_embedding_size != args.embedding_size:
                mismatched_params.append(
                    f"Embedding Size (loaded {loaded_embedding_size}, current {args.embedding_size})")
            if loaded_hidden_size is not None and loaded_hidden_size != args.hidden_size:
                mismatched_params.append(
                    f"Hidden Size (loaded {loaded_hidden_size}, current {args.hidden_size})")
            if loaded_padding_idx != PADDING_IDX:
                mismatched_params.append(
                    f"Padding Index (loaded {loaded_padding_idx}, current {PADDING_IDX})")

            if mismatched_params:
                print(
                    "\nWarning: Mismatch between loaded model parameters and current arguments:")
                for param in mismatched_params:
                    print(f"  - {param}")
                print(
                    "Proceeding with loaded model structure, but this might lead to issues.\n")
                # Re-initialize model with loaded parameters if there's a mismatch
                model = IthemalRNN(vocab_size=loaded_vocab_size or vocab_size,  # Prioritize loaded if available
                                   embedding_size=loaded_embedding_size or args.embedding_size,
                                   hidden_size=loaded_hidden_size or args.hidden_size,
                                   padding_idx=loaded_padding_idx).to(device)

            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state loaded successfully.")

            # Optionally load optimizer state and starting epoch
            if 'optimizer_state_dict' in checkpoint and not args.reset_optimizer:
                try:
                    optimizer.load_state_dict(
                        checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded successfully.")
                except ValueError as e:
                    print(
                        f"Warning: Could not load optimizer state, possibly due to parameter mismatch: {e}. Starting with a fresh optimizer.")
            if 'epoch' in checkpoint and not args.reset_epoch:
                start_epoch = checkpoint['epoch']  # Start from the next epoch
                print(f"Resuming training from epoch {start_epoch + 1}")
            if 'val_loss' in checkpoint:
                best_val_loss = checkpoint['val_loss']
                print(
                    f"Loaded previous best validation loss: {best_val_loss:.4f}")

        else:
            print(
                f"Warning: Checkpoint file not found at {args.load_model}. Training from scratch.")
    else:
        print("No checkpoint specified. Training from scratch.")

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # --- Training Loop ---
    print(f"Starting training from epoch {start_epoch + 1}...")
    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):  # Start loop from start_epoch
        model.train()
        epoch_train_loss = 0.0
        processed_train_samples = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for batch in progress_bar:
            if batch is None:
                continue

            targets = batch['targets'].to(device)

            optimizer.zero_grad()

            try:
                predictions = model(batch)

                loss = criterion(predictions, targets)

                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item() * targets.size(0)
                processed_train_samples += targets.size(0)
                progress_bar.set_postfix(
                    batch_loss=f'{loss.item():.4f}')

            except Exception as e:
                print(f"\nError during training batch processing: {e}")
                import traceback
                traceback.print_exc()
                continue

        avg_epoch_train_loss = epoch_train_loss / \
            processed_train_samples if processed_train_samples > 0 else 0
        progress_bar.close()

        # --- Validation Phase ---
        model.eval()
        epoch_val_loss = 0.0
        processed_val_samples = 0
        val_progress_bar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)

        with torch.no_grad():
            for batch in val_progress_bar:
                if batch is None:
                    continue

                targets = batch['targets'].to(device)

                try:
                    predictions = model(batch)
                    loss = criterion(predictions, targets)
                    epoch_val_loss += loss.item() * targets.size(0)
                    processed_val_samples += targets.size(0)
                except Exception as e:
                    print(f"\nError during validation batch processing: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        avg_epoch_val_loss = epoch_val_loss / \
            processed_val_samples if processed_val_samples > 0 else 0
        val_progress_bar.close()

        # --- Epoch Summary ---
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Avg Training Loss: {avg_epoch_train_loss:.4f}")
        print(f"  Avg Validation Loss: {avg_epoch_val_loss:.4f}")
        print(f"  Time Elapsed: {elapsed_time:.2f}s")
        print("-" * 50)

        # --- Save Model Checkpoint ---
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
                    'loss': avg_epoch_train_loss,
                    'val_loss': avg_epoch_val_loss,
                    'vocab_size': vocab_size,
                    'embedding_size': args.embedding_size,
                    'hidden_size': args.hidden_size,
                    'padding_idx': PADDING_IDX,
                    'target_arch': TARGET,
                    'args': args
                }, save_file)
                print(
                    f"New best model saved to {save_file} (Val Loss: {best_val_loss:.4f})")

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
                'padding_idx': PADDING_IDX,
                'target_arch': TARGET,
                'args': args
            }, save_file_epoch)
            print(f"Epoch {epoch+1} checkpoint saved to {save_file_epoch}")

    print("Training finished.")
    end_time = time.time()
    print(f"Total Training Time: {(end_time - start_time):.2f}s")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Train Ithemal model on BHive data for TARGET={TARGET}.")
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Number of samples per batch')
    parser.add_argument('--deserialize', action='store_true', default=False,
                        help='Choose to load data from serialized files')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--embedding-size', type=int, default=256,
                        help='Size of token embeddings')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Size of LSTM hidden states')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--save-path', type=str, default='trained_models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save-interval', type=int, default=0,
                        help='Save a checkpoint every N epochs (0 to disable, only best model saved)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training even if available')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to a pre-trained model checkpoint to load and continue training')
    parser.add_argument('--reset-optimizer', action='store_true', default=False,
                        help='Do not load optimizer state from checkpoint, start fresh')
    parser.add_argument('--reset-epoch', action='store_true', default=False,
                        help='Do not resume epoch count from checkpoint, start from epoch 0')

    args = parser.parse_args()

    if not os.path.exists(THROUGHPUT_FILE):
        print(f"Error: Derived throughput file not found at {THROUGHPUT_FILE}")
        print(
            f"Ensure the TARGET environment variable ('{TARGET}') is correct and the file exists.")
        sys.exit(1)

    train(args)
