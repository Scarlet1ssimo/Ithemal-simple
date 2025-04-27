from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_cost import DataItem


class IthemalRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_classes=1):
        super(IthemalRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.embedding = nn.Embedding(628, self.embedding_size)
        self.token_rnn = nn.LSTM(self.embedding_size, self.hidden_size)
        self.instr_rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, item: DataItem):
        token_output_map = {}
        token_state_map = {}
        for instr, token_inputs in zip(item.block.instrs, item.x):
            token_state = self.get_init()
            tokens = self.final_embeddings(
                torch.LongTensor(token_inputs)).unsqueeze(1)
            output, state = self.token_rnn(tokens, token_state)
            token_output_map[instr] = output
            token_state_map[instr] = state

        instr_chain = torch.stack([token_output_map[instr][-1]
                                  for instr in item.block.instrs])
        return self.pred_of_instr_chain(instr_chain)

    def pred_of_instr_chain(self, instr_chain):
        _, final_state_packed = self.instr_rnn(
            instr_chain, self.get_init())
        final_state = final_state_packed[0]

        return self.linear(final_state.squeeze()).squeeze()

    def get_token_init(self):
        return nn.Parameter(torch.zeros(
            1, 1, self.hidden_size, requires_grad=True))


if __name__ == '__main__':
    # Example usage
    model = IthemalRNN(256, 256)
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
