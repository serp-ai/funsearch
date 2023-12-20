# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

import numpy as np

from implementation import evaluator
from implementation import programs_database
from implementation.assistant import Assistant


class LLM:
    """Language model that predicts continuation of provided source code."""

    def __init__(
        self,
        samples_per_prompt: int,
        endpoint: str,
        tokenizer: str = "mistralai/Mixtral-8x7B-v0.1",
        max_seq_len: int = 32768,
        max_generated_tokens: int = 32768,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        top_p_delta: float = 0.0,
        temperature_delta: float = 0.0,
        stop: list | None = None,
    ) -> None:
        self._samples_per_prompt = samples_per_prompt
        self.assistant = Assistant(
            endpoint=endpoint, tokenizer=tokenizer, max_seq_len=max_seq_len
        )
        self.temperature = temperature
        self.temperature_delta = temperature_delta
        self.top_k = top_k
        self.top_p = top_p
        self.top_p_delta = top_p_delta
        self.max_generated_tokens = max_generated_tokens
        self.stop = stop

    def _draw_sample(self, prompt: str) -> str:
        """Returns a predicted continuation of `prompt`."""
        temperature = self.temperature
        if self.temperature_delta > 0:
            temperature += np.random.uniform(
                -self.temperature_delta, self.temperature_delta
            )
        top_p = self.top_p
        if self.top_p_delta > 0:
            top_p += np.random.uniform(-self.top_p_delta, self.top_p_delta)
        sample = self.assistant.get_response(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=self.top_k,
            stop=self.stop,
            max_tokens=self.max_generated_tokens,
        )
        return sample

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
    """Node that samples program continuations and sends them for analysis."""

    def __init__(
        self,
        database: programs_database.ProgramsDatabase,
        evaluators: Sequence[evaluator.Evaluator],
        samples_per_prompt: int,
        endpoint: str,
        tokenizer: str = "mistralai/Mixtral-8x7B-v0.1",
        max_seq_len: int = 32768,
        max_generated_tokens: int = 32768,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        top_p_delta: float = 0.0,
        temperature_delta: float = 0.0,
        stop: list | None = None,
    ) -> None:
        self._database = database
        self._evaluators = evaluators
        self._llm = LLM(
            samples_per_prompt,
            endpoint,
            tokenizer,
            max_seq_len,
            max_generated_tokens,
            temperature,
            top_k,
            top_p,
            top_p_delta,
            temperature_delta,
            stop,
        )

    def sample(self):
        """Continuously gets prompts, samples programs, sends them for analysis."""
        while True:
            prompt = self._database.get_prompt()
            samples = self._llm.draw_samples(prompt.code)
            # This loop can be executed in parallel on remote evaluator machines.
            for sample in samples:
                chosen_evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample, prompt.island_id, prompt.version_generated
                )
