"""Shared helpers for tests that need OIDC tokens signed with a throwaway key."""

import time

import jwt

ISSUER = "https://oidc.vercel.com/test-team"
AUDIENCE = "https://vercel.com/test-team"
SUBJECT = "owner:test-team:project:test-project:environment:production"
PREVIEW_SUBJECT = "owner:test-team:project:test-project:environment:preview"


class FakeSigningKey:
    def __init__(self, key):
        self.key = key


class FakeJwksClient:
    def __init__(self, public_key):
        self.public_key = public_key

    def get_signing_key_from_jwt(self, _token):
        return FakeSigningKey(self.public_key)


def make_token(key, **overrides):
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": SUBJECT,
        "iat": now,
        "nbf": now,
        "exp": now + 600,
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test"})
