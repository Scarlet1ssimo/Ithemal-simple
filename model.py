from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_cost import DataItem


class IthemalRNN(nn.Module):
    # Add vocab_size parameter
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes=1):
        super(IthemalRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.vocab_size = vocab_size # Store vocab_size

        # Use vocab_size for embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.token_rnn = nn.LSTM(self.embedding_size, self.hidden_size)
        self.instr_rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    # Add get_init method
    def get_init(self, batch_size=1):
        """Returns initial hidden and cell states for LSTMs."""
        # The dimensions should match (num_layers * num_directions, batch_size, hidden_size)
        # Assuming 1 layer, 1 direction
        # Use requires_grad=True if you want to learn the initial state
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        cell = torch.zeros(1, batch_size, self.hidden_size)
        return (hidden, cell)

    def forward(self, item: DataItem):
        # Check if item.x is empty or None
        if not item.x:
            # Handle empty blocks: return a zero tensor or raise an error
            # Returning zero might be suitable if loss handles it
            print(f"Warning: Encountered empty block (Code ID: {item.code_id}). Returning zero prediction.")
            return torch.zeros(self.num_classes) # Match output dimension

        token_output_map = {}
        token_state_map = {}
        # Initialize hidden state for token RNN (batch_size=1)
        token_init_state = self.get_init(batch_size=1)

        for instr, token_inputs in zip(item.block.instrs, item.x):
            # Check if token_inputs is empty
            if not token_inputs:
                # Handle instructions with no tokens (shouldn't happen with valid tokenizer output)
                # Use a zero vector or a special embedding
                print(f"Warning: Instruction with no tokens (Code ID: {item.code_id}). Using zero vector.")
                # Create a zero tensor matching the expected output shape of token_rnn
                # Shape: (seq_len, batch, hidden_size) -> (1, 1, hidden_size)
                token_output_map[instr] = torch.zeros(1, 1, self.hidden_size)
                continue # Skip RNN for this instruction

            # Ensure token_inputs are valid indices
            valid_token_inputs = [idx for idx in token_inputs if 0 <= idx < self.vocab_size]
            if len(valid_token_inputs) != len(token_inputs):
                print(f"Warning: Invalid token indices found (Code ID: {item.code_id}). Clamping/Skipping invalid tokens.")
                if not valid_token_inputs:
                    # Handle case where all tokens are invalid
                    print(f"Warning: Instruction with only invalid tokens (Code ID: {item.code_id}). Using zero vector.")
                    token_output_map[instr] = torch.zeros(1, 1, self.hidden_size)
                    continue
                token_inputs = valid_token_inputs

            # Embed tokens
            tokens = self.embedding(torch.LongTensor(token_inputs)).unsqueeze(1)

            # Pass through token RNN
            # Use token_init_state for each instruction independently
            output, state = self.token_rnn(tokens, token_init_state)
            token_output_map[instr] = output
            # token_state_map[instr] = state # Not typically needed unless state is passed between instructions

        # Check if token_output_map is empty (e.g., block had instructions but all were invalid/empty)
        if not token_output_map:
            print(f"Warning: Block resulted in no valid instruction outputs (Code ID: {item.code_id}). Returning zero prediction.")
            return torch.zeros(self.num_classes)

        # Stack the last output state of each instruction's token sequence
        # Ensure we only stack valid outputs
        instr_chain_list = [token_output_map[instr][-1] for instr in item.block.instrs if instr in token_output_map]
        if not instr_chain_list:
             print(f"Warning: Block resulted in no valid instruction outputs after filtering (Code ID: {item.code_id}). Returning zero prediction.")
             return torch.zeros(self.num_classes)

        instr_chain = torch.stack(instr_chain_list)

        # Initialize hidden state for instruction RNN (batch_size=1)
        instr_init_state = self.get_init(batch_size=1)
        _, final_state_packed = self.instr_rnn(instr_chain, instr_init_state)
        final_state = final_state_packed[0] # Get hidden state (h_n)

        # Pass final instruction RNN state through linear layer
        return self.linear(final_state.squeeze(0)).squeeze(-1) # Adjust squeeze dims if needed

    # Remove the old get_token_init method
    # def get_token_init(self):
    #     return nn.Parameter(torch.zeros(
    #         1, 1, self.hidden_size, requires_grad=True))


if __name__ == '__main__':
    # Example usage
    # Define dummy vocab size for testing structure
    dummy_vocab_size = 628 # Or load metadata to get actual size
    model = IthemalRNN(dummy_vocab_size, 256, 256)
    print("-" * 50)
    print("Model Structure:")
    print(model)
    print("-" * 50)

    # Calculate and print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # Calculate and print trainable parameters
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,}")
    print("-" * 50)
