

## install.py
`python install.py` should handle
- install dependencies
- download & preprocess dataset (if applicable)

Note:
- `torch` should not be a dependency, it has to work with latest pytorch master.

## hubconf.py
`hubconf.py` should contain a `class Model` with the following structure:

```
class Model:
    def __init__(self, device=None, jit=False):
        """ Required """
        self.device = device
        self.jit = jit

    def get_module(self):
        """ Required
        Returns `model`, `example_input`
        `model` should be torchscript model if self.jit is True.
        Both `model` and `example_input` should be on `self.device` properly.
        Both `model` and `example_input` can be lists/dicts as long as it's
        consumable by train/eval function. """

    def train(self):
        """ Required
        Runs training on `model` with `example_input` from get_modules()"""

    def eval(self):
        """ Optional
        Runs evaluation on `model` with `example_input` from get_modules()"""
        raise NotImplementedError()
```
To test locally before submitting a PR, add the following snippet to `hubconf.py` and run `python hubconf.py`.
```
if __name__ == '__main__':
    m = Model()
    m.get_module()
    m.train()
    m.eval()  # This line is optional
```

### Questions
* example_input doesn't have to be in certain data structure. It can be anything that f takes.
* Speed difference between CPU/CUDA


### Notes
* Everytime you want to include new changes in submodules, use `git submodule update --remote -f` to update hash in this repo.
* CI is tested agaisnt pytorch nightly package from conda

