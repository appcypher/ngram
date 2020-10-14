import numpy as np
import re
from functools import reduce

class Model:
    """
    A word gram model.
    """

    def __init__(self, context_window=2):
        self.token_windows = {}
        self.context_window = context_window
        self.sample_size = 0

    def tokenize(self, sample):
        return [token.lower() for token in re.findall(r"\w+|[',.]", sample)]

    def train(self, sample):
        tokens = self.tokenize(sample)
        self.create_token_windows(tokens)
        print("Token Windows Frequency>", self.token_windows)

    def create_token_windows(self, tokens):
        token_length = len(tokens)
        num_iterations = token_length - self.context_window

        for i in range(num_iterations):
            token_window = "*".join(tokens[i : i + self.context_window])
            next_token = tokens[i + self.context_window] if (i != num_iterations - 1) else ""

            if not self.token_windows.get(token_window):
                self.token_windows[token_window] = { next_token: 1 }
            elif not self.token_windows[token_window].get(next_token):
                self.token_windows[token_window][next_token] = 1
            else:
                self.token_windows[token_window][next_token] += 1

        self.sample_size = len(self.token_windows)

    def test(self, sample):
        tokens = self.tokenize(sample)

        distribution = []

        if token_window := self.token_windows.get("*".join(tokens)):
            sample_size = reduce(lambda a, b: a + b, token_window.values())

            for key, val in token_window.items():
                distribution.append((key, val / sample_size))

        return distribution
