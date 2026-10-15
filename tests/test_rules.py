import pytest
import torch

from mr_nwm.intent import IntentTokenizer, intent_batch
from mr_nwm.rules import compile_objectives


@pytest.mark.parametrize("split", ["train", "test", "composition"])
def test_lexicon_is_a_strong_control_for_restricted_grammar(split):
    tokens, targets = intent_batch(256, torch.Generator().manual_seed(91), split)
    torch.testing.assert_close(compile_objectives(tokens), targets, rtol=0, atol=0)


def test_unsupported_intent_is_rejected_by_rule_compiler():
    with pytest.raises(ValueError, match="no supported"):
        compile_objectives(IntentTokenizer().encode(["please optimize unsupported"]))
