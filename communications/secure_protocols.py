import os
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import pqcrypto.kem.kyber512

class QuantumResistantKeyExchange:
    """
    Implements a quantum-resistant key exchange using Kyber (NIST PQC finalist).
    """
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        pk, sk = pqcrypto.kem.kyber512.generate_keypair()
        return pk, sk

    def encapsulate(self, pk: bytes) -> Tuple[bytes, bytes]:
        ciphertext, shared_secret = pqcrypto.kem.kyber512.encrypt(pk)
        return ciphertext, shared_secret

    def decapsulate(self, ciphertext: bytes, sk: bytes) -> bytes:
        return pqcrypto.kem.kyber512.decrypt(ciphertext, sk)

class SecureChannel:
    """
    Provides authenticated, quantum-resistant encrypted communication using AES-GCM with PQ key exchange.
    """
    def __init__(self, shared_secret: bytes):
        # Derive key for symmetric encryption
        self.key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'starstorm-secure-comm',
        ).derive(shared_secret)
        self.aesgcm = AESGCM(self.key)

    def encrypt(self, plaintext: bytes, aad: bytes = b'') -> Tuple[bytes, bytes]:
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, plaintext, aad)
        return nonce, ciphertext

    def decrypt(self, nonce: bytes, ciphertext: bytes, aad: bytes = b'') -> bytes:
        return self.aesgcm.decrypt(nonce, ciphertext, aad)
