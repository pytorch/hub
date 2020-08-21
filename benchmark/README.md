## Running Benchmarks
There are currently two top-level scripts for running the models in hub.  

`test.py` offers the simplest wrapper around the infrastructure for iterating through each model in the hub and installing and executing it.

test_bench.py is a pytest-benchmark script that leverages the same infrastructure but collects benchmark statistics and supports filtering ala pytest.  

In each model repo, the assumption is that the user would already have all of the torch family of packages installed (torch, torchtext, torchvision, ...) but it installs the rest of the dependencies for the model.

### Using `test.py`
`python test.py` will execute the setup and run steps for each model in the hub.

Note: setup steps require connectivity, make sure to enable a proxy if needed.

### Using pytest-benchmark driver
Run `python test.py --setup_only` first to cause setup steps for each model to happen.

`pytest test_bench.py` invokes the benchmark driver.  See `--help` for a complete list of options.  

Some useful options include
- `--benchmark-autosave` (or other save related flags) to get .json output
- `-k <filter expression>` (standard pytest filtering)
- `--collect-only` only show what tests would run, useful to see what models there are or debug your filter expression

## Nightly CI runs
Currently, hub models run on nightly pytorch builds and push data to scuba.  

See [Unidash](https://www.internalfb.com/intern/unidash/dashboard/pytorch_benchmarks/hub_detail/) (internal only)

## Adding new models
Instructions for adding new models are currently under development in a quip.  At a high level, each model exists in its own repository, usually forked from an original open source repository and modified to add `install.py` and `hubconf.py` files which enable the hub scripts to interact with a known API.
