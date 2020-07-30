import os
import pytest
import torch
from bench_utils import workdir, setup, list_model_paths

def pytest_generate_tests(metafunc, display_len=12):
    # This is where the list of models to test can be configured
    # e.g. by using info in metafunc.config
    all_models = list_model_paths()
    short_names = []
    for name in all_models:
        short = os.path.split(name)[1]
        if len(short) > display_len:
           short = short[:display_len] + "..."
        short_names.append(short)
    metafunc.parametrize('model_path', all_models, ids=short_names, scope="module")
    metafunc.parametrize('device', ['cpu', 'cuda'], scope='module')

def setup_module(module):
    """Run setup steps for hub models.
    """
    setup()

@pytest.fixture(scope='module')
def hub_model(request, model_path, device):
    """Constructs a model object for pytests to use.
    Any pytest function that consumes a 'modeldef' arg will invoke this
    automatically, and reuse it for each test that takes that combination
    of arguments within the module.

    If reusing the module between tests isn't safe, change 'scope' parameter.
    """
    install_file = 'install.py'
    hubconf_file = 'hubconf.py'
    with workdir(model_path):
        hub_module = torch.hub.import_module(hubconf_file, hubconf_file)
        Model = getattr(hub_module, 'Model', None)
        if not Model:
            raise RuntimeError('Missing class Model in {}/hubconf.py'.format(model_path))
        return Model(device=device)


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=True,
)
class TestBenchNetwork:
    """
    This test class will get instantiated once for each 'model_stuff' provided
    by the fixture above, for each device listed in the device parameter.
    """
    def test_train(self, hub_model, benchmark):
        benchmark(hub_model.train)

    def test_eval(self, hub_model, benchmark):
        benchmark(hub_model.eval)
