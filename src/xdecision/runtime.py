"""Load a checkpoint or complete GGUF with a PyTorch or native MLX backend.

Q8_0 is dequantized on load; this is not a native quantized GGML backend.
"""
import tempfile
from pathlib import Path
import laya
from .gguf_io import restore_gguf


class Model:
    def __init__(self, path, device=None, *, backend='torch', dtype=None, batch_size=16):
        if backend not in ('torch', 'mlx'):
            raise ValueError('backend must be torch or mlx')
        self._temporary = None
        if backend == 'mlx':
            from .mlx_runtime import MLXModel
            self._model = MLXModel(path, device, dtype=dtype or 'float16', batch_size=batch_size)
            return
        if dtype is not None:
            raise ValueError('dtype is only supported by the MLX entry point')
        try:
            restored = Path(path)
            if not restored.is_dir():
                self._temporary = tempfile.TemporaryDirectory(prefix='xdecision-')
                restored = Path(self._temporary.name)/'model'
                restore_gguf(path, None, restored)
            self._model = laya.load(str(restored), device=device or 'cpu')
        except BaseException:
            if self._temporary is not None:
                self._temporary.cleanup()
            raise

    def predict(self, state, questions):
        if self._model is None:
            raise RuntimeError('Model is closed')
        result = self._model.predict(state, questions)
        result['model'] = 'xDecision'
        return result

    def close(self):
        if hasattr(self._model, 'close'):
            self._model.close()
        self._model = None
        if self._temporary is not None:
            self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load(path, device=None, *, backend='torch', dtype=None, batch_size=16):
    return Model(path, device, backend=backend, dtype=dtype, batch_size=batch_size)
