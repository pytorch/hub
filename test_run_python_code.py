import pytest
import glob


ALL_FILES = glob.glob("python_code/*.py")


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

    exec(open(file_path).read())
