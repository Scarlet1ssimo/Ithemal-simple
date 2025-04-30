import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import argparse
import time

# Import necessary components from other files
import data_cost as dt
from bhive_dataloader import BHiveDataset, collate_fn
from model import IthemalRNN

# Ensure ITHEMAL_HOME is set correctly (copied from bhive_dataloader.py)
if 'ITHEMAL_HOME' not in os.environ:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ithemal_dir = os.path.abspath(os.path.join(script_dir, 'Ithemal'))
    if os.path.isdir(ithemal_dir):
        os.environ['ITHEMAL_HOME'] = ithemal_dir
        print(f"Setting ITHEMAL_HOME to: {ithemal_dir}")
    else:
        print("Warning: ITHEMAL_HOME environment variable not set and Ithemal directory not found automatically.")

if 'ITHEMAL_HOME' not in os.environ or not os.path.isdir(os.environ['ITHEMAL_HOME']):
     print("Error: ITHEMAL_HOME is not set or points to an invalid directory.")
     sys.exit(1)

def train(args):
    """Main training loop."""

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading Ithemal metadata...")
    try:
        ithemal_data_ref = dt.DataInstructionEmbedding()
        ithemal_data_ref.read_meta_data()
        vocab_size = len(ithemal_data_ref.token_to_hot_idx)
        print(f"Metadata loaded. Vocabulary size: {vocab_size}")
    except Exception as e:
        print(f"Failed to load Ithemal metadata: {e}")
        sys.exit(1)

    print(f"Loading dataset from: {args.throughput_file}")
    dataset = BHiveDataset(throughput_file=args.throughput_file, token_idx_map_ref=ithemal_data_ref)
    # Consider splitting data into train/validation sets if needed
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # For now, use the full dataset for training demonstration
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    print(f"Dataset loaded. Number of samples: {len(dataset)}")

    # --- Model Initialization ---
    print("Initializing model...")
    model = IthemalRNN(vocab_size=vocab_size,
                       embedding_size=args.embedding_size,
                       hidden_size=args.hidden_size).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # --- Loss and Optimizer ---
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Using Loss: {criterion}")
    print(f"Using Optimizer: Adam (lr={args.lr})")

    # --- Training Loop ---
    print("Starting training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        processed_batches = 0
        processed_items = 0

        for i, batch_data in enumerate(train_loader):
            if batch_data is None: # Skip empty batches
                continue

            batch_loss = 0.0
            valid_items_in_batch = 0
            optimizer.zero_grad() # Zero gradients for the batch

            # Process each item in the batch individually
            for item in batch_data:
                # Move target tensor to the correct device
                target = item.y.to(device)

                # Forward pass for a single item
                try:
                    # Ensure model and item data are on the same device if needed
                    # (DataItem structure might not be directly movable, handle tensors inside)
                    prediction = model(item)
                    prediction = prediction.to(device) # Ensure prediction is on the correct device

                    # Calculate loss for this item
                    loss = criterion(prediction, target)

                    # Accumulate loss for backpropagation (average over batch later)
                    # Scale loss by 1/batch_size for averaging before backward()
                    # Or accumulate and divide before optimizer.step()
                    batch_loss += loss
                    valid_items_in_batch += 1
                    processed_items += 1

                except Exception as e:
                    print(f"\nError processing item (Code ID {item.code_id}, Hex: {dataset.data[item.code_id][0][:20] if hasattr(item, 'code_id') and item.code_id < len(dataset.data) else 'N/A'}...): {e}")
                    import traceback
                    traceback.print_exc()
                    # Optionally continue to the next item or stop

            # Backpropagate and update weights if any valid items were processed
            if valid_items_in_batch > 0:
                # Average the loss over the valid items in the batch
                avg_batch_loss = batch_loss / valid_items_in_batch
                avg_batch_loss.backward() # Calculate gradients based on average loss
                optimizer.step() # Update weights

                epoch_loss += avg_batch_loss.item() * valid_items_in_batch # Accumulate total epoch loss
                processed_batches += 1

                # Print progress
                if (i + 1) % args.log_interval == 0:
                    print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{len(train_loader)}], Avg Batch Loss: {avg_batch_loss.item():.4f}")

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / processed_items if processed_items > 0 else 0
        elapsed_time = time.time() - start_time
        print("-" * 50)
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Average Training Loss: {avg_epoch_loss:.4f}")
        print(f"  Time Elapsed: {elapsed_time:.2f}s")
        print("-" * 50)

        # --- Optional: Validation Step ---
        # if val_loader:
        #     model.eval() # Set model to evaluation mode
        #     val_loss = 0.0
        #     with torch.no_grad():
        #         for batch_data in val_loader:
        #              if batch_data is None: continue
        #              for item in batch_data:
        #                  target = item.y.to(device)
        #                  prediction = model(item).to(device)
        #                  loss = criterion(prediction, target)
        #                  val_loss += loss.item()
        #     avg_val_loss = val_loss / len(val_dataset) # Adjust denominator if filtering Nones
        #     print(f"  Validation Loss: {avg_val_loss:.4f}")
        #     print("-" * 50)

        # --- Optional: Save Model Checkpoint ---
        if args.save_path and (epoch + 1) % args.save_interval == 0:
            save_file = os.path.join(args.save_path, f"ithemal_bhive_epoch_{epoch+1}.pt")
            os.makedirs(args.save_path, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                # Add other relevant info like vocab_size, args
                'vocab_size': vocab_size,
                'embedding_size': args.embedding_size,
                'hidden_size': args.hidden_size,
            }, save_file)
            print(f"Model checkpoint saved to {save_file}")

    print("Training finished.")
    end_time = time.time()
    print(f"Total Training Time: {(end_time - start_time):.2f}s")

    # --- Optional: Save Final Model ---
    if args.save_path:
        save_file = os.path.join(args.save_path, "ithemal_bhive_final.pt")
        os.makedirs(args.save_path, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
             # Add other relevant info
            'vocab_size': vocab_size,
            'embedding_size': args.embedding_size,
            'hidden_size': args.hidden_size,
            'args': args
        }, save_file)
        print(f"Final model saved to {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ithemal model on BHive data.")
    parser.add_argument('--throughput-file', type=str, required=True,
                        help='Path to the BHive throughput CSV file (e.g., bhive/benchmark/throughput/skl.csv)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, # Effective batch size is handled in the loop
                        help='Number of samples to process before optimizer step')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--embedding-size', type=int, default=256,
                        help='Size of token embeddings')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Size of LSTM hidden states')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of dataloader workers (set to 0 if subprocess causes issues)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='How many batches to wait before logging training status')
    parser.add_argument('--save-path', type=str, default='trained_models',
                        help='Directory to save model checkpoints and final model')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='How many epochs to wait before saving a model checkpoint')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training even if available')

    args = parser.parse_args()

    # Validate throughput file path
    if not os.path.exists(args.throughput_file):
        print(f"Error: Throughput file not found at {args.throughput_file}")
        sys.exit(1)

    train(args)