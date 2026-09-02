# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.attention.backends.flash_attn import _get_split_dcp_context_window


@pytest.mark.parametrize(
    ("max_query_len", "expected"),
    [
        (1, [2046, 0]),
        (2, [2045, 1]),
        (7, [2040, 6]),
    ],
)
def test_split_dcp_context_window_uniform_decode(max_query_len, expected):
    assert _get_split_dcp_context_window(
        [2047, 0],
        causal=True,
        max_query_len=max_query_len,
        num_query_tokens=4 * max_query_len,
        num_reqs=4,
        num_prefill_reqs=0,
    ) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"causal": False},
        {"num_prefill_reqs": 1},
        {"num_query_tokens": 27},
        {"max_query_len": 2048, "num_query_tokens": 8192},
    ],
)
def test_split_dcp_context_window_keeps_unsupported_layouts(kwargs):
    params = {
        "causal": True,
        "max_query_len": 7,
        "num_query_tokens": 28,
        "num_reqs": 4,
        "num_prefill_reqs": 0,
    }
    params.update(kwargs)
    window = [2047, 0]
    assert _get_split_dcp_context_window(window, **params) is window


def test_split_dcp_context_window_disabled():
    assert (
        _get_split_dcp_context_window(
            None,
            causal=True,
            max_query_len=7,
            num_query_tokens=28,
            num_reqs=4,
            num_prefill_reqs=0,
        )
        is None
    )
