"""Load a complete GGUF through the reference PyTorch decision runtime.

Q8_0 is dequantized on load; this is not a native quantized GGML backend.
"""
import tempfile
from pathlib import Path
import laya
from .gguf_io import restore_gguf


class Model:
    def __init__(self, path, device='cpu'):
        self._temporary = tempfile.TemporaryDirectory(prefix='xdecision-')
        try:
            restored = Path(self._temporary.name)/'model'
            restore_gguf(path, None, restored)
            self._model = laya.load(str(restored), device=device)
        except BaseException:
            self._temporary.cleanup()
            raise

    def predict(self, state, questions):
        result = self._model.predict(state, questions)
        result['model'] = 'xDecision'
        return result

    def close(self):
        self._model = None
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load(path, device='cpu'):
    return Model(path, device)
