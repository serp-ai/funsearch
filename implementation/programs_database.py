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

"""A programs database that implements the evolutionary algorithm."""
import sqlite3
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any

from absl import logging
import numpy as np
import scipy

from implementation import code_manipulation
from implementation import config as config_lib

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f"`logits` contains non-finite value(s): {non_finites}")
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1 :])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    return scores_per_test[list(scores_per_test.keys())[-1]]


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
    code: str
    version_generated: int
    island_id: int


class ProgramsDatabase:
    def __init__(
        self,
        config: config_lib.ProgramsDatabaseConfig,
        template: code_manipulation.Program,
        function_to_evolve: str,
        db_path="programs.db",
    ):
        self._config = config
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._db_path = db_path
        self._init_db()

        self._islands = [
            Island(
                template,
                function_to_evolve,
                config.functions_per_prompt,
                config.cluster_sampling_temperature_init,
                config.cluster_sampling_temperature_period,
                db_path,
            )
            for _ in range(config.num_islands)
        ]

        self._best_score_per_island = [-float("inf")] * config.num_islands
        self._best_program_per_island = [None] * config.num_islands
        self._best_scores_per_test_per_island = [None] * config.num_islands
        self._last_reset_time = time.time()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS programs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT,
                    island_id INTEGER,
                    score REAL,
                    signature TEXT
                )
            """
            )

    def get_prompt(self) -> Prompt:
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)

    def _register_program_in_island(
        self,
        program: code_manipulation.Function,
        island_id: int,
        scores_per_test: ScoresPerTest,
    ):
        self._islands[island_id].register_program(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info("Best score of island %d increased to %s", island_id, score)

    def register_program(
        self,
        program: code_manipulation.Function,
        island_id: int | None,
        scores_per_test: ScoresPerTest,
    ):
        if island_id is None:
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, scores_per_test)
        else:
            self._register_program_in_island(program, island_id, scores_per_test)

        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()

    def reset_islands(self):
        indices_sorted_by_score = np.argsort(
            self._best_score_per_island
            + np.random.randn(len(self._best_score_per_island)) * 1e-6
        )
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period,
                self._db_path,
            )
            self._best_score_per_island[island_id] = -float("inf")
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id, founder_scores)


class Island:
    def __init__(
        self,
        template: code_manipulation.Program,
        function_to_evolve: str,
        functions_per_prompt: int,
        cluster_sampling_temperature_init: float,
        cluster_sampling_temperature_period: int,
        db_path: str,
    ):
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._functions_per_prompt = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period
        self._db_path = db_path

        self._clusters = {}
        self._num_programs = 0

    def register_program(
        self, program: code_manipulation.Function, scores_per_test: ScoresPerTest
    ):
        signature = _get_signature(scores_per_test)
        score = _reduce_score(scores_per_test)
        code = str(program)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO programs (code, island_id, score, signature) VALUES (?, ?, ?, ?)",
                (code, id(self), score, str(signature)),
            )

        if signature not in self._clusters:
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1

    def get_prompt(self) -> tuple[str, int]:
        signatures = list(self._clusters.keys())
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures]
        )
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
            1 - (self._num_programs % period) / period
        )
        probabilities = _softmax(cluster_scores, temperature)
        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)
        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities
        )
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)
        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(
        self, implementations: Sequence[code_manipulation.Function]
    ) -> str:
        implementations = copy.deepcopy(implementations)
        versioned_functions = []
        for i, implementation in enumerate(implementations):
            new_function_name = f"{self._function_to_evolve}_v{i}"
            implementation.name = new_function_name
            if i >= 1:
                implementation.docstring = (
                    f"Improved version of `{self._function_to_evolve}_v{i - 1}`."
                )
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name
            )
            versioned_functions.append(
                code_manipulation.text_to_function(implementation)
            )
        next_version = len(implementations)
        new_function_name = f"{self._function_to_evolve}_v{next_version}"
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body="",
            docstring=f"Improved version of `{self._function_to_evolve}_v{next_version - 1}`.",
        )
        versioned_functions.append(header)
        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        return str(prompt)


class Cluster:
    """A cluster of programs on the same island and with the same Signature."""

    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        """Reduced score of the signature that this cluster represents."""
        return self._score

    def register_program(self, program: code_manipulation.Function) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorther programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
            max(self._lengths) + 1e-6
        )
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)
