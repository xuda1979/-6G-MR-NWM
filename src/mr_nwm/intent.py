"""Small auditable intent corpus, with held-out wording and a held-out composition.

This is supervised parsing of a controlled vocabulary, not open-language zero-shot
reasoning. Only training phrases contribute words to the tokenizer vocabulary.
"""

import re

import torch

OBJECTIVES = ("throughput", "latency", "energy")
TRAIN_COMBINATIONS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))
TRAIN_TEMPLATES = ("prioritize {}", "please optimize {}", "optimize {} for this cell")
TEST_TEMPLATES = ("for this cell prioritize {} please", "please prioritize {}")


def phrase(combination, template):
    return template.format(" and ".join(OBJECTIVES[index] for index in combination))


class IntentTokenizer:
    def __init__(self):
        words = set()
        for combination in TRAIN_COMBINATIONS:
            for template in TRAIN_TEMPLATES:
                words.update(self.words(phrase(combination, template)))
        self.vocabulary = {"<pad>": 0, "<unk>": 1}
        self.vocabulary.update({word: i + 2 for i, word in enumerate(sorted(words))})

    @staticmethod
    def words(text):
        return re.findall(r"[a-z]+", text.lower())

    def encode(self, texts, length=12):
        output = torch.zeros(len(texts), length, dtype=torch.long)
        for row, text in enumerate(texts):
            words = self.words(text)
            if not words or len(words) > length:
                raise ValueError("intent must be nonempty and fit the tokenizer length")
            output[row, : len(words)] = torch.tensor(
                [self.vocabulary.get(word, 1) for word in words]
            )
        return output


def intent_batch(size, generator, split="train"):
    if split not in {"train", "test", "composition"}:
        raise ValueError("unknown intent split")
    combinations = ((0, 1, 2),) if split == "composition" else TRAIN_COMBINATIONS
    templates = TRAIN_TEMPLATES if split == "train" else TEST_TEMPLATES
    texts, weights = [], torch.zeros(size, 3)
    choices = torch.randint(len(combinations), (size,), generator=generator)
    wording = torch.randint(len(templates), (size,), generator=generator)
    for row, (choice, template) in enumerate(zip(choices.tolist(), wording.tolist())):
        combination = combinations[choice]
        texts.append(phrase(combination, templates[template]))
        weights[row, list(combination)] = 1 / len(combination)
    return IntentTokenizer().encode(texts), weights
