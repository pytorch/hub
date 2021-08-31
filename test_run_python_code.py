from subprocess import check_output, STDOUT, CalledProcessError
import sys
import pytest
import glob


PYTHON_CODE_DIR = "python_code"
ALL_FILES = glob.glob(PYTHON_CODE_DIR + "/*.py")


@pytest.mark.parametrize('file_path', ALL_FILES)
def test_run_file(file_path):
    if 'nvidia' in file_path:
        # FIXME: NVIDIA models checkoints are on cuda
        pytest.skip("temporarily disabled")
    if 'pytorch_fairseq_translation' in file_path:
        pytest.skip("temporarily disabled")
    if 'ultralytics_yolov5' in file_path:
        # FIXME torch.nn.modules.module.ModuleAttributeError: 'autoShape' object has no attribute 'fuse
        pytest.skip("temporarily disabled")
    if 'huggingface_pytorch-transformers' in file_path:
        # FIXME torch.nn.modules.module.ModuleAttributeError: 'autoShape' object has no attribute 'fuse
        pytest.skip("temporarily disabled")
    if 'pytorch_fairseq_roberta' in file_path:
        pytest.skip("temporarily disabled")

    # We just run the python files in a separate sub-process. We really want a
    # subprocess here because otherwise we might run into package versions
    # issues: imagine script A that needs torchvivion 0.9 and script B that
    # needs torchvision 0.10. If script A is run prior to script B in the same
    # process, script B will still be run with torchvision 0.9 because the only
    # "import torchvision" statement that counts is the first one, and even
    # torchub sys.path shenanigans can do nothing about this. By creating
    # subprocesses we're sure that all file executions are fully independent.
    try:
        # This is inspired (and heavily simplified) from
        # https://github.com/cloudpipe/cloudpickle/blob/343da119685f622da2d1658ef7b3e2516a01817f/tests/testutils.py#L177
        out = check_output([sys.executable, file_path], stderr=STDOUT)
        print(out.decode())
    except CalledProcessError as e:
        raise RuntimeError(f"Script {file_path} errored with output:\n{e.output.decode()}")
