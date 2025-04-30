import subprocess

import pandas as pd
import os
import tqdm
from multiprocessing import Pool, cpu_count

from torch.utils.data import Dataset


class BHiveDataset(Dataset):

    # Define a static method for disassembly to be used by multiprocessing
    @staticmethod
    def _disassemble_hex(hex_val):
        try:
            # Ensure the path to disasm is correct relative to where the script is run
            # Or provide an absolute path if necessary
            return subprocess.check_output(
                ["./bhive/benchmark/disasm", hex_val]).decode('utf8')
        except Exception as e:
            print(f"Error disassembling {hex_val}: {e}")
            return None # Return None or handle error appropriately

    # Add first_n parameter with a default value of None
    def __init__(self, root_path: str, disasm: bool = False, first_n: int = None):
        self.data = pd.read_csv(os.path.join(
            root_path, 'benchmark/categories.csv'), header=None, names=['hex', 'category'])
        self.cpu_type_list = ['hsw', 'ivb', 'skl']
        for cpu_type in self.cpu_type_list:
            throughput = pd.read_csv(os.path.join(
                root_path, f'benchmark/throughput/{cpu_type}.csv'), header=0, names=['hex', f"throughput_{cpu_type}"])
            self.data = pd.merge(self.data, throughput, on='hex', how='right')
        self.data.dropna(inplace=True)

        # Limit the number of records if first_n is specified
        if first_n is not None and first_n > 0:
            self.data = self.data.head(first_n).copy() # Use .copy() to avoid SettingWithCopyWarning later

        # Reset index after potential slicing and dropna
        self.data.reset_index(drop=True, inplace=True)

        self.data['asm'] = None
        if disasm:
            print('Disassembling hex to asm in parallel...')
            hex_list = self.data['hex'].tolist()

            # Use multiprocessing Pool
            # Use cpu_count() for potentially better performance, adjust if needed
            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                # Use tqdm with pool.imap for progress bar, or pool.map without progress
                # imap preserves order and is generally memory-efficient for large iterables
                results = list(tqdm.tqdm(pool.imap(self._disassemble_hex, hex_list), total=len(hex_list)))

            # Assign the results list directly to the 'asm' column
            self.data['asm'] = results
            # Optionally, handle any None results if errors occurred during disassembly
            # e.g., self.data.dropna(subset=['asm'], inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = {
            'category': self.data['category'].values[index],
            'hex': self.data['hex'].values[index],
            'asm': self.data['asm'].values[index]
        }
        label = {cpu_type: float(
            self.data[f'throughput_{cpu_type}'].values[index]) for cpu_type in self.cpu_type_list}
        return data, label


if __name__ == '__main__':
    # Example usage with first_n
    dataset = BHiveDataset('./bhive', disasm=True, first_n=100) # Load only the first 100 records
    print(f"Loaded {len(dataset)} records.")
    print(dataset.data.head()) # Print head to verify
    if len(dataset) > 0:
        print(dataset[0])
    else:
        print("Dataset is empty.")

    # Example usage without first_n (loads all)
    # dataset_full = BHiveDataset('./bhive', disasm=False)
    # print(f"\nLoaded {len(dataset_full)} records (full dataset).")
