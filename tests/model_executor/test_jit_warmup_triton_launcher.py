# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

from vllm.model_executor.warmup.jit_warmup import WarmupChoices, _when
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DeclarativeTritonJitKernel,
    LaunchSpec,
    VllmTritonJitKernel,
    compile_key,
    kernel_launcher,
)


class _FakeTritonKernel:
    arg_names = ("first", "second", "CONST")

    def __init__(self) -> None:
        self.warmup_calls: list[dict[str, Any]] = []

    def warmup(self, **kwargs: Any) -> None:
        self.warmup_calls.append(kwargs)


class _TestTritonKernel(VllmTritonJitKernel["_TestTritonKernel.CompileKey"]):
    kernel = _FakeTritonKernel()

    @dataclass(frozen=True)
    class CompileKey:
        value: int

    def dispatch(self, *, value: int) -> CompileKey:
        return self.CompileKey(value=value)

    def get_warmup_keys(self) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(value=1)

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(first="warmup", second=compile_key.value, runtime_launcher=None)

    @kernel_launcher
    def __call__(
        self,
        first: str,
        second: int,
        runtime_launcher: Any,
    ) -> LaunchSpec:
        return (2,), dict(
            CONST=7,
            _runtime_launcher=runtime_launcher,
            _runtime_launcher_arg_count=2,
        )


def test_triton_launcher_supports_compile_and_runtime_adapters() -> None:
    owner = _TestTritonKernel()
    owner.kernel.warmup_calls.clear()

    owner.compile(owner.CompileKey(value=1))
    assert owner.kernel.warmup_calls == [
        {"grid": (1,), "first": "warmup", "second": 1, "CONST": 7}
    ]

    runtime_calls: list[tuple[Any, ...]] = []

    def runtime_launcher(kernel: Any, grid: Any, *args: Any, **kwargs: Any) -> None:
        runtime_calls.append((kernel, grid, args, kwargs))

    owner("runtime", 2, runtime_launcher)
    assert runtime_calls == [(owner.kernel, (2,), ("runtime", 2), {"CONST": 7})]


def test_triton_launcher_supports_cpu_function_wrappers() -> None:
    calls: list[tuple[Any, ...]] = []

    def kernel(first: str, second: int, CONST: int) -> None:
        calls.append((first, second, CONST))

    class FuncWrapper:
        def __init__(self) -> None:
            self.func = kernel

        def __getitem__(self, _grid: Any) -> Any:
            return self.func

    class TestCpuKernel(_TestTritonKernel):
        kernel: Any = FuncWrapper()

    owner = TestCpuKernel()

    owner("runtime", 2, None)
    assert calls == [("runtime", 2, 7)]


def test_declarative_triton_kernel_replays_and_deduplicates_cases() -> None:
    class TestDeclarativeKernel(DeclarativeTritonJitKernel):
        kernel = _FakeTritonKernel()

        def warmup_cases(self) -> tuple[dict[str, Any], ...]:
            case = dict(
                CONST=compile_key(7),
                first="warmup",
                second=1,
                runtime_launcher=None,
            )
            return case, case

        def launch_spec(
            self,
            first: str,
            second: int,
            runtime_launcher: Any,
        ) -> LaunchSpec:
            return (2,), dict(
                CONST=7,
                _runtime_launcher=runtime_launcher,
                _runtime_launcher_arg_count=2,
            )

    owner = TestDeclarativeKernel()
    keys = owner.get_warmup_keys()

    assert len(keys) == 1
    assert keys[0].values == (("CONST", 7),)
    assert owner.warmup_inputs(keys[0]) == {
        "first": "warmup",
        "runtime_launcher": None,
        "second": 1,
    }

    owner.compile(keys[0])
    assert owner.kernel.warmup_calls == [
        {"grid": (1,), "first": "warmup", "second": 1, "CONST": 7}
    ]


def test_declarative_triton_kernel_traces_ast_cases() -> None:
    class TestDeclarativeKernel(DeclarativeTritonJitKernel):
        kernel = _FakeTritonKernel()

        def warmup_cases(
            self,
        ) -> dict[str, Any]:
            value = WarmupChoices(2, 3)
            _when(value > 1)
            return dict(
                dtype=compile_key("bf16"),
                first="warmup",
                second=value,
                runtime_launcher=None,
            )

        def launch_spec(
            self,
            first: str,
            second: int,
            runtime_launcher: Any,
        ) -> LaunchSpec:
            return (2,), dict(
                CONST=7,
                _runtime_launcher=runtime_launcher,
                _runtime_launcher_arg_count=2,
            )

    owner = TestDeclarativeKernel()
    keys = owner.get_warmup_keys()

    assert len(keys) == 1
    assert keys[0].values == (("dtype", "bf16"),)
    assert keys[0].inputs.as_dict()["dtype"] == "bf16"
    assert "dtype" not in owner.warmup_inputs(keys[0])
    assert owner.warmup_inputs(keys[0])["second"] == 2

    owner.kernel.warmup_calls.clear()
    owner.compile(keys[0])
    assert owner.kernel.warmup_calls == [
        {"grid": (1,), "first": "warmup", "second": 2, "CONST": 7}
    ]


def test_declarative_triton_kernel_inherits_runtime_binding() -> None:
    class TestDeclarativeKernel(DeclarativeTritonJitKernel):
        kernel = _FakeTritonKernel()

        def warmup_cases(self) -> dict[str, Any]:
            return dict(
                first="warmup",
                second=compile_key(1),
                runtime_launcher=None,
            )

        def launch_spec(
            self,
            first: str,
            second: int,
            runtime_launcher: Any,
        ) -> LaunchSpec:
            return (2,), dict(
                CONST=7,
                _runtime_launcher=runtime_launcher,
                _runtime_launcher_arg_count=2,
            )

    owner = TestDeclarativeKernel()
    owner.kernel.warmup_calls.clear()
    key = owner.get_warmup_keys()[0]
    assert key.values == (("second", 1),)
    owner.compile(key)
    assert owner.kernel.warmup_calls == [
        {"grid": (1,), "first": "warmup", "second": 1, "CONST": 7}
    ]

    runtime_calls: list[tuple[Any, ...]] = []

    def runtime_launcher(kernel: Any, grid: Any, *args: Any, **kwargs: Any) -> None:
        runtime_calls.append((kernel, grid, args, kwargs))

    owner("runtime", 2, runtime_launcher=runtime_launcher)
    assert runtime_calls == [(owner.kernel, (2,), ("runtime", 2), {"CONST": 7})]


def test_compute_slot_mapping_uses_named_launcher_inputs(monkeypatch) -> None:
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.block_table import ComputeSlotMappingKernel

    owner = ComputeSlotMappingKernel()
    compile_key = owner.CompileKey(
        kv_cache_block_size=16,
        blocks_per_kv_block=1,
        total_cp_world_size=2,
        total_cp_rank=1,
        cp_kv_cache_interleave_size=1,
        block_table_stride=128,
        block_size=16,
    )
    launches: list[tuple[Any, ...]] = []

    def launch(launch_spec: Any, inputs: Any) -> None:
        grid, kwargs = launch_spec
        launches.append((grid, inputs, kwargs))

    monkeypatch.setattr(owner, "launch", launch)
    owner.compile(compile_key)

    grid, inputs, kwargs = launches[0]
    assert grid == (2,)
    assert inputs["num_tokens"] == 2
    assert inputs["block_table_stride"] == 128
    assert kwargs == {
        "KV_CACHE_BLOCK_SIZE": 16,
        "BLOCKS_PER_KV_BLOCK": 1,
        "TOTAL_CP_WORLD_SIZE": 2,
        "TOTAL_CP_RANK": 1,
        "CP_KV_CACHE_INTERLEAVE_SIZE": 1,
        "PAD_ID": PAD_SLOT_ID,
        "BLOCK_SIZE": owner.triton_block_size,
    }
