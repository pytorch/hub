"""test.py
Setup and Run hub models.

Make sure to enable an https proxy if necessary, or the setup steps may hang.
"""
# This file shows how to use the benchmark suite from user end.
import argparse
import time
from bench_utils import workdir, setup, list_models

def run_model(model_class, model_path):
    for device in ('cpu', 'cuda'):
        print('Running [{}] on device: {}'.format(str(model_path), device))
        m = model_class(device=device)

        try:
            module, example_inputs = m.get_module()
            module(*example_inputs)
        except NotImplementedError:
            print('Method get_module is not implemented, skipping...')

        try:
            start = time.time()
            m.train()
            print('Finished training on device: {} in {}s.'.format(device, time.time() - start))
        except NotImplementedError:
            print('Method train is not implemented, skipping...')

        try:
            start = time.time()
            m.eval()
            print('Finished eval on device: {} in {}s.'.format(device, time.time() - start))
        except NotImplementedError:
            print('Method eval is not implemented, skipping...')


def run_models():
    for model_class, model_path in list_models():
        with workdir(model_path):
            run_model(model_class, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup_only', action='store_true',
                        help='run setup steps only, then exit.')
    args = parser.parse_args()

    setup()

    if not args.setup_only:
        run_models()
