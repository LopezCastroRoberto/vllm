# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    GatherInitialStatesKernel,
)


def test_gather_initial_states_warmup_keys_and_launch() -> None:
    kernel = GatherInitialStatesKernel()
    keys = kernel.get_warmup_keys(
        row_size=8192,
        state_dtype=torch.bfloat16,
        indices_dtype=torch.int32,
        launch_pdl=False,
    )

    assert len(keys) == 4
    inputs_by_key = [kernel.warmup_inputs(key) for key in keys]
    assert {inputs["indices"].aligned for inputs in inputs_by_key} == {True, False}
    assert {
        inputs["has_initial_state"].aligned for inputs in inputs_by_key
    } == {True, False}
    assert {inputs["row_size"] for inputs in inputs_by_key} == {8192}
    assert {inputs["block_size"] for inputs in inputs_by_key} == {1024}

    inputs = kernel.warmup_inputs(keys[0])
    grid, launch_kwargs = kernel.launch_spec(**inputs)
    assert grid == (8, 1)
    assert launch_kwargs["row_size"] == 8192
    assert launch_kwargs["BLOCK_SIZE"] == 1024
