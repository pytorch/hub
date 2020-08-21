import os
from pathlib import Path
import subprocess
import sys
import torch
from urllib import request

git_submodule_suggestion = "Have you run\n`" \
                           "`git submodule update --init --recursive`?"
proxy_suggestion = "Unable to verify https connectivity, " \
                   "required for setup.\n" \
                   "Do you need to use a proxy?"

this_dir = Path(__file__).parent.absolute()
model_dir = 'models/'
install_file = 'install.py'
hubconf_file = 'hubconf.py'


def _test_https(test_url='https://github.com', timeout=0.5):
    try:
        request.urlopen(test_url, timeout=timeout)
    except OSError:
        return False
    return True


def _install_deps(model_path):
    if os.path.exists(os.path.join(model_path, install_file)):
        subprocess.check_call([sys.executable, install_file], cwd=model_path)
    else:
        print('No install.py is found in {}.'.format(model_path))
        print(git_submodule_suggestion)
        sys.exit(-1)


class workdir():
    def __init__(self, path):
        self.path = path
        self.cwd = os.getcwd()

    def __enter__(self):
        sys.path.insert(0, self.path)
        os.chdir(self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            os.chdir(self.cwd)
            sys.path.remove(self.path)
        except ValueError:
            pass


def list_model_paths():
    p = Path(__file__).parent.joinpath(model_dir)
    return [str(child.absolute()) for child in p.iterdir()]


def setup():
    if not _test_https():
        print(proxy_suggestion)
        sys.exit(-1)

    _install_deps(this_dir)
    for model_path in list_model_paths():
        _install_deps(model_path)


def list_models():
    models = []
    for model_path in list_model_paths():
        with workdir(model_path):
            try:
                hub_module = torch.hub.import_module(hubconf_file, hubconf_file)
                Model = getattr(hub_module, 'Model', None)
            except FileNotFoundError:
                raise RuntimeError(f"Unable to find {hubconf_file} in {model_path}.\n"
                                   "{git_submodule_suggestion")
            models.append(Model)
    return zip(models, list_model_paths())
