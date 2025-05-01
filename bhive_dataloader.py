import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import subprocess
import binascii
import sys
import data_cost as dt
import utilities as ut
import pickle
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils  # Add this import

if 'ITHEMAL_HOME' not in os.environ or not os.path.isdir(os.environ['ITHEMAL_HOME']):
    print("Error: ITHEMAL_HOME is not set or points to an invalid directory.")
    sys.exit(1)

_TOKENIZER = os.path.join(
    os.environ['ITHEMAL_HOME'], 'data_collection', 'build', 'bin', 'tokenizer')
_fake_intel = '\n'*500  # From test.py


class BHiveDataset(Dataset):
    def __init__(self, throughput_file: str, token_idx_map_ref: dt.DataInstructionEmbedding, deserialize=False, dump=False):
        """
        Args:
            throughput_file (string): Path to the BHive throughput CSV file (e.g., skl.csv).
            ithemal_data_ref (dt.DataInstructionEmbedding): Reference Ithemal data object for metadata.
        """
        self.deserialize = deserialize
        self.new_embedding = False
        if token_idx_map_ref:
            self.token_idx_map = token_idx_map_ref
        else:
            self.token_idx_map = dt.DataInstructionEmbedding()
            self.token_idx_map.read_meta_data()
            self.new_embedding = True
        self.data = []
        try:
            # BHive throughput files have no header, columns are hex_string, throughput
            if self.deserialize:
                # Deserialize the data
                with open(f"dataset_dump_{TARGET}.pkl", 'rb') as f:
                    self.data = pickle.load(f)
                print(f"Loaded {len(self.data)} samples from dataset dump.")
                return
            df = pd.read_csv(throughput_file, header=None,
                             names=['hex', 'throughput'])
            df['throughput'] = df['throughput'] / 100  # Normalize throughput
            df.dropna(inplace=True)  # Drop any NaN values
            self.data = df[['hex', 'throughput']].values.tolist()
            print(f"Loaded {len(self.data)} samples from {throughput_file}")
            if dump:
                for i, (hex_string, throughput) in tqdm(enumerate(self.data)):
                    try:
                        # Use datum_of_code logic (adapted from test.py)
                        datum = self._datum_of_code_internal(
                            hex_string, verbose=False)  # verbose=False for training
                        # Set the target value (throughput)
                        datum.y = torch.tensor(throughput, dtype=torch.float32)
                        self.data[i] = datum
                    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError, ValueError) as e:
                        print(
                            f"Warning: Skipping sample {i} (hex: {hex_string[:20]}...) due to error: {e}")
                        # Return None or raise an error that the collate_fn can handle
                        return None
                    except Exception as e:
                        print(
                            f"Warning: Skipping sample {i} (hex: {hex_string[:20]}...) due to unexpected error: {e}")
                        import traceback
                        traceback.print_exc()
                        return None
                with open(f"dataset_dump_{TARGET}.pkl", 'wb') as f:
                    pickle.dump(self.data, f)
        except FileNotFoundError:
            print(f"Error: Throughput file not found at {throughput_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading throughput file {throughput_file}: {e}")
            sys.exit(1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dt.DataItem:
        if self.deserialize:
            return self.data[idx]
        hex_string, throughput = self.data[idx]

        try:
            # Use datum_of_code logic (adapted from test.py)
            datum = self._datum_of_code_internal(
                hex_string, verbose=False)  # verbose=False for training
            # Set the target value (throughput)
            datum.y = torch.tensor(throughput, dtype=torch.float32)
            return datum
        except (FileNotFoundError, PermissionError, subprocess.CalledProcessError, ValueError) as e:
            print(
                f"Warning: Skipping sample {idx} (hex: {hex_string[:20]}...) due to error: {e}")
            # Return None or raise an error that the collate_fn can handle
            return None
        except Exception as e:
            print(
                f"Warning: Skipping sample {idx} (hex: {hex_string[:20]}...) due to unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _datum_of_code_internal(self, block_hex, verbose):
        """Internal version of datum_of_code from test.py"""
        if not os.path.exists(_TOKENIZER):
            raise FileNotFoundError(f"Tokenizer not found at {_TOKENIZER}.")
        if not os.access(_TOKENIZER, os.X_OK):
            raise PermissionError(
                f"Tokenizer at {_TOKENIZER} is not executable.")

        try:
            process_token = subprocess.Popen(
                [_TOKENIZER, block_hex, '--token'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_token, stderr_token = process_token.communicate()
            if process_token.returncode != 0:
                raise subprocess.CalledProcessError(
                    process_token.returncode, process_token.args, output=stdout_token, stderr=stderr_token)
            xml = stdout_token.decode('utf-8', errors='ignore')
        except subprocess.CalledProcessError as e:
            # print(f"Error running tokenizer (--token) for hex {block_hex}: {e.stderr.decode('utf-8', errors='ignore')}")
            raise  # Re-raise the error to be caught by __getitem__

        # We don't need intel assembly for training, use fake one
        intel = _fake_intel

        # Use a temporary copy of the data object to avoid race conditions if using multiprocessing
        if self.new_embedding:
            self.token_idx_map.raw_data = [(-1, -1, intel, xml)]
            self.token_idx_map.data = []
            self.token_idx_map.prepare_data(fixed=False, progress=False)
            if not self.token_idx_map.data:
                raise ValueError("prepare_data did not produce any DataItem.")

            return self.token_idx_map.data[-1]
        else:
            return self.token_idx_map.process_data([(-1, -1, intel, xml)])[-1]


def collate_fn(batch):
    """
    Collate function to handle variable length sequences and None items.
    Filters out None items resulting from errors in __getitem__.
    Pads token sequences (x) and instruction sequences.
    """
    # Filter out None items
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # Return None if the batch is empty after filtering

    # Sort batch by number of instructions (required for packing) - descending order
    batch.sort(key=lambda item: len(item.block.instrs), reverse=True)

    # --- Prepare data for batching ---
    targets = torch.tensor([item.y.item() for item in batch], dtype=torch.float32)
    instruction_lengths = torch.tensor([len(item.block.instrs) for item in batch], dtype=torch.long)
    max_instr_len = instruction_lengths[0].item()  # Max instructions in this batch

    # --- Pad token sequences ---
    # We need to pad tokens for *all* instructions across the batch
    all_token_sequences = []
    token_lengths = []  # Length of each instruction's token sequence
    instr_boundaries = [0]  # Track where each basic block's instructions start in the flattened list
    current_boundary = 0
    for item in batch:
        for instr_tokens in item.x:
            all_token_sequences.append(torch.tensor(instr_tokens, dtype=torch.long))
            token_lengths.append(len(instr_tokens))
        current_boundary += len(item.block.instrs)
        instr_boundaries.append(current_boundary)

    # Pad the token sequences for all instructions
    padded_tokens = rnn_utils.pad_sequence(all_token_sequences, batch_first=True, padding_value=0)  # Assuming 0 is the padding index
    token_lengths = torch.tensor(token_lengths, dtype=torch.long)

    return {
        'padded_tokens': padded_tokens,  # (total_instrs_in_batch, max_token_len)
        'token_lengths': token_lengths,  # (total_instrs_in_batch,)
        'instruction_lengths': instruction_lengths,  # (batch_size,) - how many instructions per item
        'instr_boundaries': instr_boundaries,  # To reconstruct instruction outputs per block
        'targets': targets  # (batch_size,)
    }


# Example usage (optional, for testing the dataloader)
def create_map():
    dataset = BHiveDataset(
        throughput_file=throughput_file_path, token_idx_map_ref=None)

    print("Token_idx_map before processing:")
    print(dataset.token_idx_map.dump_dataset_params())
    # Create DataLoader instance
    # Using batch_size > 1 with our collate_fn means the batch is a list of DataItems.
    # The training loop needs to handle this list.
    # num_workers > 0 might cause issues with subprocess/shared state. Start with 0.
    dataloader = DataLoader(dataset, batch_size=256, collate_fn=collate_fn,
                            shuffle=False, num_workers=0)

    print(f"Dataset size: {len(dataset)}")

    # Iterate through a few batches
    for _ in tqdm(dataloader):
        continue

    print("\nDataloader test finished.")
    print("Token_idx_map after processing:")
    print(dataset.token_idx_map.dump_dataset_params())

    output_file = f'vocab_map_{TARGET}.pkl'

    with open(output_file, 'wb') as f:
        pickle.dump(dataset.token_idx_map.dump_dataset_params(), f)


def dump():
    dataset = BHiveDataset(
        throughput_file=throughput_file_path, token_idx_map_ref=dt.load_token_idx_map(f'vocab_map_{TARGET}.pkl'), deserialize=False, dump=True)


def test_dataloader():
    dataset = BHiveDataset(
        throughput_file=throughput_file_path, token_idx_map_ref=dt.load_token_idx_map(f'vocab_map_{TARGET}.pkl'), deserialize=True)

    print("Token_idx_map before processing:")
    print(dataset.token_idx_map.dump_dataset_params())
    # Create DataLoader instance
    # Using batch_size > 1 with our collate_fn means the batch is a list of DataItems.
    # The training loop needs to handle this list.
    # num_workers > 0 might cause issues with subprocess/shared state. Start with 0.
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn,
                            shuffle=True, num_workers=8)

    print(f"Dataset size: {len(dataset)}")

    # Iterate through a few batches
    for i, batch_data in enumerate(dataloader):
        if batch_data is None:  # Handle empty batches after filtering Nones
            print(f"Batch {i+1}: Skipped (empty after filtering errors)")
            continue

        print(f"\nBatch {i+1} (Size: {len(batch_data)}):")
        batch_loss = 0  # Example: Accumulate loss within the batch
        for j, item in enumerate(batch_data):
            print(f"  Item {j}:")
            print(f"    Code ID (Dummy): {item.code_id}")
            print(f"    Throughput (y): {item.y.item():.4f}")
            print(f"    Num Instructions: {len(item.block.instrs)}")
            # Example: Simulate getting a prediction and calculating loss for this item
            # prediction = model(item) # Assuming model processes one item
            # loss = criterion(prediction, item.y)
            # batch_loss += loss.item()

        # print(f"  Avg Batch Loss (Simulated): {batch_loss / len(batch_data):.4f}")

        if i >= 2:  # Print first 3 batches
            break

    print("\nDataloader test finished.")
    # print("Token_idx_map after processing:")
    # print(dataset.token_idx_map.dump_dataset_params())


TARGET = os.environ.get("ITHEMAL_TARGET", "skl")
if __name__ == '__main__':
    print("Testing BHive Dataloader...")

    # Specify the path to a BHive throughput file
    # Make sure this path is correct relative to where you run the script
    throughput_file_path = os.path.join(
        # Example for Skylake
        'bhive', 'benchmark', 'throughput', f'{TARGET}.csv')

    if not os.path.exists(throughput_file_path):
        print(
            f"Error: Example throughput file not found: {throughput_file_path}")
        print("Please ensure the BHive submodule is present and the path is correct.")
        sys.exit(1)

    if sys.argv[1] == "create_map":
        # Create dataset instance
        create_map()
    elif sys.argv[1] == "dump":
        dump()
    else:
        # Test the dataloader with an existing token index map
        test_dataloader()
