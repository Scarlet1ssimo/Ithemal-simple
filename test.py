#!/usr/bin/env python

import argparse
import sys
import binascii
# Need to import utilities for BasicBlock and Instruction
# Adjust the path if your Ithemal directory is located differently relative to mp4
try:
    import utilities as ut
except ImportError:
    print("Error: Could not import Ithemal utilities.")
    print("Please ensure the Ithemal submodule is present and accessible.")
    sys.exit(1)
import copy
# import data.data_cost as dt # Already imported below
import itertools
import multiprocessing
import os
import subprocess
import sys
import threading
import torch
import warnings
import data_cost as dt
import binascii
START_MARKER = binascii.unhexlify('bb6f000000646790')
END_MARKER = binascii.unhexlify('bbde000000646790')

# Make sure ITHEMAL_HOME is set, or adjust the path
if 'ITHEMAL_HOME' not in os.environ:
    # Assuming Ithemal is a sibling directory or find it relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ithemal_dir = os.path.abspath(os.path.join(script_dir, 'Ithemal'))
    if os.path.isdir(ithemal_dir):
        os.environ['ITHEMAL_HOME'] = ithemal_dir
        print(f"Setting ITHEMAL_HOME to: {ithemal_dir}")
    else:
        # Fallback or error if Ithemal directory is not found
        print("Warning: ITHEMAL_HOME environment variable not set and Ithemal directory not found automatically.")
        # Provide a default or raise an error
        # os.environ['ITHEMAL_HOME'] = '/path/to/Ithemal' # Example fallback

# Check if ITHEMAL_HOME is set now
if 'ITHEMAL_HOME' not in os.environ or not os.path.isdir(os.environ['ITHEMAL_HOME']):
     print("Error: ITHEMAL_HOME is not set or points to an invalid directory.")
     sys.exit(1)

_TOKENIZER = os.path.join(
    os.environ['ITHEMAL_HOME'], 'data_collection', 'build', 'bin', 'tokenizer')

_fake_intel = '\n'*500


def load_model_and_data_(model_file):
    # This function seems unused in the original test.py,
    # and loading a model file isn't needed just to get data structure.
    # We'll stick to the original load_model_and_data function below.
    pass


def load_model_and_data(fname):
    # Simplified for testing data structure: only need DataInstructionEmbedding
    data = dt.DataInstructionEmbedding()
    # Need to read metadata to initialize token dictionaries and offsets
    try:
        data.read_meta_data()
    except FileNotFoundError as e:
        print(f"Error reading metadata: {e}")
        print(f"Ensure '{os.path.join(os.environ['ITHEMAL_HOME'], 'common/inputs')}' contains necessary files (encoding.h, offsets.txt).")
        sys.exit(1)
    # No model loading needed here for data structure inspection
    return data


def datum_of_code(data, block_hex, verbose):
    # Ensure tokenizer exists and is executable
    if not os.path.exists(_TOKENIZER):
        raise FileNotFoundError(f"Tokenizer not found at {_TOKENIZER}. Make sure Ithemal is built (run Ithemal/build_all.sh).")
    if not os.access(_TOKENIZER, os.X_OK):
         raise PermissionError(f"Tokenizer at {_TOKENIZER} is not executable. Check file permissions.")

    # Use try-except for subprocess calls
    try:
        # Use Popen for better error handling and capturing stderr
        process_token = subprocess.Popen([_TOKENIZER, block_hex, '--token'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_token, stderr_token = process_token.communicate()
        if process_token.returncode != 0:
             raise subprocess.CalledProcessError(process_token.returncode, process_token.args, output=stdout_token, stderr=stderr_token)
        xml = stdout_token.decode('utf-8', errors='ignore')

    except subprocess.CalledProcessError as e:
        print(f"Error running tokenizer (--token) for hex {block_hex}:")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        if e.output: print(f"Stdout:\n{e.output.decode('utf-8', errors='ignore')}")
        if e.stderr: print(f"Stderr:\n{e.stderr.decode('utf-8', errors='ignore')}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred running tokenizer (--token): {e}")
        raise

    if verbose:
        try:
            process_intel = subprocess.Popen([_TOKENIZER, block_hex, '--intel'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_intel, stderr_intel = process_intel.communicate()
            if process_intel.returncode != 0:
                 raise subprocess.CalledProcessError(process_intel.returncode, process_intel.args, output=stdout_intel, stderr=stderr_intel)
            intel = stdout_intel.decode('utf-8', errors='ignore')

        except subprocess.CalledProcessError as e:
            print(f"Warning: Error running tokenizer (--intel) for hex {block_hex}:")
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            if e.output: print(f"Stdout:\n{e.output.decode('utf-8', errors='ignore')}")
            if e.stderr: print(f"Stderr:\n{e.stderr.decode('utf-8', errors='ignore')}")
            # Continue with fake intel if --intel fails but --token worked
            print("Continuing with fake intel assembly.")
            intel = _fake_intel
        except Exception as e:
            print(f"An unexpected error occurred running tokenizer (--intel): {e}")
            intel = _fake_intel
    else:
        intel = _fake_intel

    # Assign to raw_data before calling prepare_data
    data.raw_data = [(-1, -1, intel, xml)] # Use -1 for dummy code_id and timing
    data.data = [] # Clear previous data if any
    # Call prepare_data to process raw_data into data.data
    # Need ut for prepare_data internal calls
    dt.ut = ut # Inject ut dependency into data_cost module
    data.prepare_data(fixed=False, progress=False) # Use fixed=False to allow new tokens

    if not data.data:
         raise ValueError("prepare_data did not produce any DataItem.")

    return data.data[-1]


def predict_raw(verbose):
    # No model needed, just the data object structure
    # Pass a dummy filename, it's not used by load_model_and_data anymore
    data = load_model_and_data("dummy_file")
    # Example hex string from BHive (first entry in categories.csv)
    # line = "554889e54157415641554154534883ec58488b05a11b2c004c8d25b61b2c004c8d2dbb1b2c00488d3dcf1b2c00488d35e31b2c0064488b042528000000488945d0488b80800000004885c0744b488b80880000004885c07441488b80900000004885c07437488b80980000004885c0742d488b80a00000004885c07423488b80a800000004885c07419488b80b00000004885c0740f488b80b80000004885c07405e931000000488b45d0488b004885c07405e91c000000488b45d0488b40084885c07405e907000000488b45d0488b40104885c07405e9f2ffffff488b45d0488b40184885c07405e9ddffffff488b45d0488b40204885c07405e9c8ffffff488b45d0488b40284885c07405e9b3ffffff488b45d0488b40304885c07405e99effffff488b45d0488b40384885c07405e989ffffff488b45d0488b40404885c07405e974ffffff488b45d0488b40484885c07405e95fffffff488b45d0488b40504885c07405e94affffff488b45d0488b40584885c07405e935ffffff488b45d0488b40604885c07405e920ffffff488b45d0488b40684885c07405e90bffffff488b45d0488b40704885c07405e9f6feffff488b45d0488b40784885c07405e9e1feffff488b45d0488b80800000004885c07405e9ccfeffff488b45d0488b80880000004885c07405e9b7feffff488b45d0488b80900000004885c07405e9a2feffff488b45d0488b809800000004885c07405e98dfeffff488b45d0488b80a00000004885c07405e978feffff488b45d0488b80a800000004885c07405e963feffff488b45d0488b80b00000004885c07405e94efeffff488b45d0488b80b800000004885c07405e939feffff488b45d031f631ff31c931d2e8868cffff488b45d04883c4585b5c415c415d415e415f5dc3"
    line = "4183ff0119c083e00885c98945c4b8010000000f4fc139c2" # Original test.py line
    try:
        datum = datum_of_code(data, line, verbose)
        
        print(datum.x)
        print(datum.y)

        print("-" * 30)
        print("Datum Object Structure:")
        print("-" * 30)

        print(f"Type: {type(datum)}")
        print(f"Code ID: {datum.code_id}")
        print(f"Timing (y): {datum.y}")

        print("\nNumerical Tokens (datum.x):")
        print("---------------------------")
        if datum.x:
            for i, instr_tokens in enumerate(datum.x):
                # Map indices back to tokens for readability
                token_names = [data.hot_idx_to_token.get(idx, f'UNK({idx})') for idx in instr_tokens]
                print(f"  Instr {i}: {instr_tokens}")
                print(f"           {token_names}")
        else:
            print("  (empty)")

        print("\nBasic Block Info (datum.block):")
        print("-------------------------------")
        if datum.block:
            print(f"  Number of Instructions: {len(datum.block.instrs)}")
            for i, instr in enumerate(datum.block.instrs):
                print(f"\n  Instruction {i}:")
                # Map opcode index back to name
                opcode_name = data.hot_idx_to_token.get(instr.opcode, f'UNK({instr.opcode})')
                print(f"    Opcode: {opcode_name} ({instr.opcode})")
                # Map operand indices back to names/types
                # Need data.sym_dict and data.mem_start from read_meta_data()
                src_names = [ut.get_name(s, data.sym_dict, data.mem_start) for s in instr.srcs]
                dst_names = [ut.get_name(d, data.sym_dict, data.mem_start) for d in instr.dsts]
                print(f"    Sources (indices): {instr.srcs}")
                print(f"    Sources (names):   {src_names}")
                print(f"    Destinations (indices): {instr.dsts}")
                print(f"    Destinations (names):   {dst_names}")
                if hasattr(instr, 'intel') and instr.intel and instr.intel != _fake_intel:
                     print(f"    Intel: {instr.intel.strip()}")
                # Print dependencies if calculated
                parent_nums = [p.num for p in instr.parents]
                child_nums = [c.num for c in instr.children]
                print(f"    Parent Instr Nums: {parent_nums}")
                print(f"    Child Instr Nums: {child_nums}")
        else:
            print("  (None)")

        print("-" * 30)


    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure Ithemal is built (run Ithemal/build_all.sh) and ITHEMAL_HOME is set correctly.")
    except PermissionError as e:
        print(f"Error: {e}")
        print("Please ensure the tokenizer is executable (check permissions).")
    except subprocess.CalledProcessError as e:
        print("Subprocess error occurred running tokenizer.")
        # Error details should have been printed within datum_of_code
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


# Run the function
if __name__ == "__main__":
    # Add verbose flag parsing if needed, default to True for this test
    predict_raw(True)