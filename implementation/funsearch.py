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
# pylint: disable=wrong-import-position
"""A single-threaded implementation of the FunSearch pipeline."""
from collections.abc import Sequence
from typing import Any
from multiprocessing import Process

import sys
import os

# append funsearch directory to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_parent_dir)

from implementation import code_manipulation  # noqa: E402
from implementation import config as config_lib  # noqa: E402
from implementation import evaluator  # noqa: E402
from implementation import programs_database  # noqa: E402
from implementation import sampler  # noqa: E402


def _extract_function_names(specification: str) -> tuple[str, str]:
    """Returns the name of the function to evolve and of the function to run."""
    run_functions = list(
        code_manipulation.yield_decorated(specification, "funsearch", "run")
    )
    if len(run_functions) != 1:
        raise ValueError("Expected 1 function decorated with `@funsearch.run`.")
    evolve_functions = list(
        code_manipulation.yield_decorated(specification, "funsearch", "evolve")
    )
    if len(evolve_functions) != 1:
        raise ValueError("Expected 1 function decorated with `@funsearch.evolve`.")
    return evolve_functions[0], run_functions[0]


def sampler_worker(sampler_instance):
    """
    Worker function for each process.
    Each process will run the `sample` method of a given Sampler instance.
    """
    sampler_instance.sample()


def main(specification: str, inputs: Sequence[Any], config: config_lib.Config):
    """Launches a FunSearch experiment."""
    function_to_evolve, function_to_run = _extract_function_names(specification)

    template = code_manipulation.text_to_program(specification)
    database = programs_database.ProgramsDatabase(
        config.programs_database, template, function_to_evolve, db_path=config.db_path
    )

    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(
            evaluator.Evaluator(
                database,
                template,
                function_to_evolve,
                function_to_run,
                inputs,
            )
        )
    # We send the initial implementation to be analysed by one of the evaluators.
    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial, island_id=None, version_generated=None)

    samplers = [
        sampler.Sampler(
            database,
            evaluators,
            config.samples_per_prompt,
            config.endpoint,
            tokenizer=config.tokenizer,
            max_seq_len=config.max_seq_len,
            max_generated_tokens=config.max_generated_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            top_p_delta=config.top_p_delta,
            temperature_delta=config.temperature_delta,
            stop=config.stop,
        )
        for _ in range(config.num_samplers)
    ]

    processes = []
    try:
        for s in samplers:
            p = Process(target=sampler_worker, args=(s,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("Interrupt received, stopping processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        sys.exit(1)


if __name__ == "__main__":
    # EXAMPLE RUN - EDIT FOR YOUR OWN USE
    with open(
        os.path.join(parent_dir, "nonsymmetric_admissible_set.txt"),
        encoding="utf-8",
    ) as f:
        specification = f.read()
    inputs = [[12, 7]]
    main(specification, inputs=inputs, config=config_lib.Config())
