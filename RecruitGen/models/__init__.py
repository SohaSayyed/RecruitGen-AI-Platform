"""RecruitGen models.

Keras runs on the PyTorch backend: PyTorch is already required by Transformer V3, so the app never
has to import TensorFlow (~20 s saved at startup). Set KERAS_BACKEND=tensorflow to switch back.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")


def skip_tensorflow_probe():
    """Keras' legacy .h5 loader imports TensorFlow just to check for TF-only config objects
    (TensorShape / TypeSpec), which our models never contain. Skip that probe on non-TF backends."""
    import keras

    if keras.backend.backend() == "tensorflow":
        return
    try:
        from keras.src.legacy.saving import json_utils

        if json_utils.tf._available is None and json_utils.tf.module is None:
            json_utils.tf._available = False
    except Exception:
        pass
