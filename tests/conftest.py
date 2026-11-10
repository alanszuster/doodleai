import os

# oneDNN's CPU kernels segfault while *training* the residual network on some machines.
# Training runs on a GPU; this only keeps the CPU smoke test of the training code stable.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa


@pytest.fixture(scope="session")
def private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)
