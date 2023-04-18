from typing import Sequence, Tuple

import torch

class PromptConverter(object):
    """ Convert Batch to Pre-train Task Needed Format
    input:    batch [(label1, sequence1), (label2, sequence2), ...]
    output:   labels, str_sequences, origin_tokens, masked_tokens(have been masked and padding), mask_ids
    """
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.pad_idx = alphabet.padding_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx

    def __call__(self, seqences: Sequence[Tuple[str, str]], prompt_toks=[]):
        batch_size = len(seqences)
        if len(prompt_toks) != 0:
            encoded_prompt = torch.tensor([self.alphabet.encode(prompt_tok)[0] for prompt_tok in prompt_toks])
        encoded_sequences = [self.alphabet.encode(sequence) for sequence in seqences]
        max_encoded_sequences_length = min(max(len(encoded_sequence) for encoded_sequence in encoded_sequences), 1022)
        tokens = torch.empty(
            (
                batch_size,
                max_encoded_sequences_length + len(prompt_toks) + 2,
            ),
            dtype=torch.int64
        )
        tokens.fill_(self.pad_idx)

        for i, encoded_sequence in enumerate(encoded_sequences):
            sequence_length = min(len(encoded_sequence), 1022)
            encoded_sequence = torch.tensor(encoded_sequence, dtype=torch.int64)
            tokens[i, 0] = self.cls_idx
            tokens[i, 1:sequence_length+1] = encoded_sequence[:sequence_length]
            tokens[i, sequence_length+1] = self.eos_idx

            if len(prompt_toks) != 0:
                tokens[i, -len(prompt_toks):] = torch.clone(encoded_prompt)

        return tokens
