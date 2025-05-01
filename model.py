from typing import NamedTuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from data_cost import DataItem


class IthemalRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes=1, padding_idx=0):
        super(IthemalRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.token_rnn = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        self.instr_rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, batch: Dict[str, Any]):
        device = self.embedding.weight.device
        batch_size = len(batch['instruction_lengths'])

        padded_tokens = batch['padded_tokens'].to(device)
        token_lengths = batch['token_lengths'].to(device)
        instruction_lengths = batch['instruction_lengths']
        targets = batch['targets'].to(device)
        instr_boundaries = batch['instr_boundaries']

        embedded_tokens = self.embedding(padded_tokens)

        packed_tokens = rnn_utils.pack_padded_sequence(embedded_tokens, token_lengths.cpu(), batch_first=True, enforce_sorted=False)

        total_instrs = padded_tokens.size(0)
        token_init_state = self.get_init(total_instrs, device)
        packed_output, (token_final_hidden, _) = self.token_rnn(packed_tokens, token_init_state)

        instr_inputs = token_final_hidden.squeeze(0)

        block_instr_inputs = []
        for i in range(batch_size):
            start = instr_boundaries[i]
            end = instr_boundaries[i+1]
            block_instr_inputs.append(instr_inputs[start:end])

        padded_instr_inputs = rnn_utils.pad_sequence(block_instr_inputs, batch_first=True, padding_value=0.0)

        packed_instr_inputs = rnn_utils.pack_padded_sequence(padded_instr_inputs, instruction_lengths.cpu(), batch_first=True, enforce_sorted=True)

        instr_init_state = self.get_init(batch_size, device)
        _, (instr_final_hidden, _) = self.instr_rnn(packed_instr_inputs, instr_init_state)

        final_block_representation = instr_final_hidden.squeeze(0)
        output = self.linear(final_block_representation).squeeze(-1)

        return output

    def get_init(self, batch_size, device):
        return (
            torch.zeros(1, batch_size, self.hidden_size, requires_grad=True, device=device),
            torch.zeros(1, batch_size, self.hidden_size, requires_grad=True, device=device),
        )


if __name__ == '__main__':
    print("Model definition updated for batch processing.")
    print("Run train.py for a functional example.")
