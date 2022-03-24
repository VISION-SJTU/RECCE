from .network import *
from .common import *

MODELS = {
    "Recce": Recce
}


def load_model(name="Recce"):
    assert name in MODELS.keys(), f"Model name can only be one of {MODELS.keys()}."
    print(f"Using model: '{name}'")
    return MODELS[name]
