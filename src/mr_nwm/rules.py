"""Exact compiler for the deliberately restricted benchmark grammar."""

import torch

from .intent import OBJECTIVES, IntentTokenizer


def compile_objectives(tokens):
    """Normalize named objectives; reject text that names none of the three labels.

    This lexicon baseline is intentionally limited to known objective words. It
    does not parse arbitrary language, negation, priorities, or numeric constraints.
    """
    vocabulary = IntentTokenizer().vocabulary
    names = torch.tensor([vocabulary[word] for word in OBJECTIVES], device=tokens.device)
    present = (tokens[..., None] == names).any(1).float()
    if (present.sum(-1) == 0).any():
        raise ValueError("no supported objective name found")
    return present / present.sum(-1, keepdim=True)
