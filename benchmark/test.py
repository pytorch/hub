# This file shows how to use the benchmark suite from user end.
import time
from utils import workdir, setup, list_models


def run_model(model_class, model_path):
    for device in ('cpu', 'cuda'):
        print('Running [{}] on device: {}'.format(str(model_path), device))
        m = model_class(device=device)

        module, example_inputs = m.get_module()
        module(*example_inputs)

        start = time.time()
        m.train()
        print('Finished training on device: {} in {}s.'.format(device, time.time() - start))

        start = time.time()
        m.eval()
        print('Finished eval on device: {} in {}s.'.format(device, time.time() - start))


def run_models():
    for model_class, model_path in list_models():
        with workdir(model_path):
            run_model(model_class, model_path)


if __name__ == '__main__':
    setup()
    run_models()
