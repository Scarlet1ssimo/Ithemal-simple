import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from tqdm import tqdm
from data import Data
import xml.etree.ElementTree as ET
import itertools
import utilities as ut
import pickle


class DataItem:

    def __init__(self, x, y, block, code_id):
        self.x = x
        self.y = y
        self.block = block
        self.code_id = code_id


class DataInstructionEmbedding(Data):

    def __init__(self):
        super(DataInstructionEmbedding, self).__init__()
        self.token_to_hot_idx = {}
        self.hot_idx_to_token = {}
        self.data = []

    def dump_dataset_params(self):
        return (self.token_to_hot_idx, self.hot_idx_to_token)

    def load_dataset_params(self, params):
        (self.token_to_hot_idx, self.hot_idx_to_token) = params

    def process_data(self, raw_data):
        def hot_idxify(elem):
            if elem not in self.token_to_hot_idx:
                raise ValueError(
                    'Ithemal does not yet support UNK tokens!')
            return self.token_to_hot_idx[elem]
        iterator = raw_data
        data = []
        for (code_id, timing, code_intel, code_xml) in iterator:
            block_root = ET.fromstring(code_xml)
            instrs = []
            raw_instrs = []
            curr_mem = self.mem_start
            for _ in range(1):  # repeat for duplicated blocks
                # handle missing or incomplete code_intel
                split_code_intel = itertools.chain(
                    (code_intel or '').split('\n'), itertools.repeat(''))
                for (instr, m_code_intel) in zip(block_root, split_code_intel):
                    raw_instr = []
                    opcode = int(instr.find('opcode').text)
                    raw_instr.extend([opcode, '<SRCS>'])
                    srcs = []
                    for src in instr.find('srcs'):
                        if src.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in src.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            srcs.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(src.text))
                            srcs.append(int(src.text))

                    raw_instr.append('<DSTS>')
                    dsts = []
                    for dst in instr.find('dsts'):
                        if dst.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in dst.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                # operands used to calculate dst mem ops are sources
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            dsts.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(dst.text))
                            dsts.append(int(dst.text))

                    raw_instr.append('<END>')
                    raw_instrs.append(list(map(hot_idxify, raw_instr)))
                    instrs.append(ut.Instruction(
                        opcode, srcs, dsts, len(instrs)))
                    instrs[-1].intel = m_code_intel

            block = ut.BasicBlock(instrs)
            block.create_dependencies()
            datum = DataItem(raw_instrs, timing, block, code_id)
            data.append(datum)
        return data

    def prepare_data(self, progress=True, fixed=False):
        def hot_idxify(elem):
            if elem not in self.token_to_hot_idx:
                if fixed:
                    # TODO: this would be a good place to implement UNK tokens
                    raise ValueError(
                        'Ithemal does not yet support UNK tokens!')
                self.token_to_hot_idx[elem] = len(self.token_to_hot_idx)
                self.hot_idx_to_token[self.token_to_hot_idx[elem]] = elem
            return self.token_to_hot_idx[elem]

        if progress:
            iterator = tqdm(self.raw_data)
        else:
            iterator = self.raw_data

        for (code_id, timing, code_intel, code_xml) in iterator:
            block_root = ET.fromstring(code_xml)
            instrs = []
            raw_instrs = []
            curr_mem = self.mem_start
            for _ in range(1):  # repeat for duplicated blocks
                # handle missing or incomplete code_intel
                split_code_intel = itertools.chain(
                    (code_intel or '').split('\n'), itertools.repeat(''))
                for (instr, m_code_intel) in zip(block_root, split_code_intel):
                    raw_instr = []
                    opcode = int(instr.find('opcode').text)
                    raw_instr.extend([opcode, '<SRCS>'])
                    srcs = []
                    for src in instr.find('srcs'):
                        if src.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in src.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            srcs.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(src.text))
                            srcs.append(int(src.text))

                    raw_instr.append('<DSTS>')
                    dsts = []
                    for dst in instr.find('dsts'):
                        if dst.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in dst.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                # operands used to calculate dst mem ops are sources
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            dsts.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(dst.text))
                            dsts.append(int(dst.text))

                    raw_instr.append('<END>')
                    raw_instrs.append(list(map(hot_idxify, raw_instr)))
                    instrs.append(ut.Instruction(
                        opcode, srcs, dsts, len(instrs)))
                    instrs[-1].intel = m_code_intel

            block = ut.BasicBlock(instrs)
            block.create_dependencies()
            datum = DataItem(raw_instrs, timing, block, code_id)
            self.data.append(datum)


def load_token_idx_map(map_savefile):
    token_idx_map = DataInstructionEmbedding()
    with open(map_savefile, 'rb') as f:
        token_to_hot_idx, hot_idx_to_token = pickle.load(f)
    token_idx_map.read_meta_data()
    token_idx_map.load_dataset_params(
        (token_to_hot_idx, hot_idx_to_token))
    return token_idx_map
