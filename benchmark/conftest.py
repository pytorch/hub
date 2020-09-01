import pytest
import torch

def pytest_addoption(parser):
    parser.addoption("--fuser", help="fuser to use for benchmarks")

def set_fuser(fuser):
    if fuser == "legacy":
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif fuser == "te":
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_bailout_depth(20)
        torch._C._jit_set_num_profiled_runs(2)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(True)

def pytest_configure(config):
    set_fuser(config.getoption("fuser"))
