from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_cost import DataItem


class IthemalRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes=1):
        super(IthemalRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.token_rnn = nn.LSTM(self.embedding_size, self.hidden_size)
        self.instr_rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, item: DataItem):
        # Determine the device from the model's parameters
        device = self.embedding.weight.device

        token_output_map = {}
        token_state_map = {}
        for instr, token_inputs in zip(item.block.instrs, item.x):
            # Ensure initial state is on the correct device
            token_state = self.get_init(device)
            # Move input tensor to the correct device before embedding
            tokens = self.embedding(
                torch.LongTensor(token_inputs).to(device)).unsqueeze(1)
            # LSTM expects input and hidden states on the same device
            output, state = self.token_rnn(tokens, token_state)
            token_output_map[instr] = output
            token_state_map[instr] = state

        # Stack the outputs (which are already on the correct device)
        instr_chain = torch.stack([token_output_map[instr][-1]
                                  for instr in item.block.instrs])
        # Ensure initial state for the second LSTM is also on the correct device
        return self.pred_of_instr_chain(instr_chain, device)

    def pred_of_instr_chain(self, instr_chain, device):
        # Ensure initial state is on the correct device
        _, final_state_packed = self.instr_rnn(
            instr_chain, self.get_init(device))
        final_state = final_state_packed[0]

        # Linear layer input and weights are already on the correct device
        return self.linear(final_state.squeeze()).squeeze()

    # Modify get_init to accept and use the device
    def get_init(self, device):
        return (
            # Create parameters directly on the specified device
            torch.zeros(1, 1, self.hidden_size,
                        requires_grad=True, device=device),
            torch.zeros(1, 1, self.hidden_size,
                        requires_grad=True, device=device),
        )


if __name__ == '__main__':
    # Example usage
    model = IthemalRNN(628, 256, 256)
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
