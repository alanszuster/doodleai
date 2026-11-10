# Auth settings are read per request, so the tests set them with monkeypatch (see the
# `client` fixture) and must not touch os.environ at import time: that would leak
# into tests/test_app.py, which sets its own AI_API_KEY.
import base64
import hashlib
import hmac
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from oidc import AUDIENCE, ISSUER, PREVIEW_SUBJECT, SUBJECT, FakeJwksClient, make_token

import auth
from app import app

PROTECTED = "/get_random_word"  # needs auth, answers 200 or 503 once authorised


@pytest.fixture(scope="module")
def other_private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def client(monkeypatch, private_key):
    monkeypatch.setenv("OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("OIDC_ALLOWED_SUBJECTS", SUBJECT)
    monkeypatch.setenv("AI_API_KEY", "legacy-test-key")
    monkeypatch.setenv("AUTH_ALLOW_API_KEY", "true")
    monkeypatch.setattr(
        auth, "_get_jwks_client", lambda _issuer: FakeJwksClient(private_key.public_key())
    )
    app.config["TESTING"] = True
    app.config["RATELIMIT_ENABLED"] = False
    with app.test_client() as test_client:
        yield test_client


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def authorised(response):
    return response.status_code in (200, 503)


def test_valid_token_is_accepted(client, private_key):
    assert authorised(client.get(PROTECTED, headers=bearer(make_token(private_key))))


def test_no_credentials_is_rejected(client):
    response = client.get(PROTECTED)
    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"


def test_health_stays_public(client):
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize(
    "overrides",
    [
        {"aud": "https://vercel.com/someone-else"},
        {"iss": "https://oidc.vercel.com/someone-else"},
        {"sub": PREVIEW_SUBJECT},
        {"sub": "owner:test-team:project:another-project:environment:production"},
        {"exp": int(time.time()) - 3600},
        {"exp": None},
        {"iat": None},
        {"sub": None},
    ],
)
def test_invalid_claims_are_rejected(client, private_key, overrides):
    token = make_token(private_key, **overrides)
    assert client.get(PROTECTED, headers=bearer(token)).status_code == 401


def test_token_signed_with_another_key_is_rejected(client, other_private_key):
    token = make_token(other_private_key)
    assert client.get(PROTECTED, headers=bearer(token)).status_code == 401


def test_hs256_key_confusion_is_rejected(client, private_key):
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    forged = _forge_hs256(public_pem, int(time.time()))
    assert client.get(PROTECTED, headers=bearer(forged)).status_code == 401


def _forge_hs256(secret, now):
    # PyJWT refuses to sign HS256 with a PEM key, so build the token by hand,
    # like an attacker would.
    def b64(data):
        return base64.urlsafe_b64encode(data).rstrip(b"=")

    header = b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": "test"}).encode())
    payload = b64(
        json.dumps(
            {
                "iss": ISSUER,
                "aud": AUDIENCE,
                "sub": SUBJECT,
                "iat": now,
                "exp": now + 600,
            }
        ).encode()
    )
    signing_input = header + b"." + payload
    signature = b64(hmac.new(secret, signing_input, hashlib.sha256).digest())
    return (signing_input + b"." + signature).decode()


def test_alg_none_is_rejected(client):
    now = int(time.time())
    unsigned = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": SUBJECT, "iat": now, "exp": now + 600},
        key=None,
        algorithm="none",
    )
    assert client.get(PROTECTED, headers=bearer(unsigned)).status_code == 401


def test_garbage_token_is_rejected(client):
    assert client.get(PROTECTED, headers=bearer("not.a.jwt")).status_code == 401


def test_oidc_unconfigured_rejects_tokens(client, private_key, monkeypatch):
    monkeypatch.delenv("OIDC_ISSUER")
    token = make_token(private_key)
    assert client.get(PROTECTED, headers=bearer(token)).status_code == 401


def test_oidc_without_allowed_subjects_rejects_tokens(client, private_key, monkeypatch):
    monkeypatch.setenv("OIDC_ALLOWED_SUBJECTS", "")
    token = make_token(private_key)
    assert client.get(PROTECTED, headers=bearer(token)).status_code == 401


def test_legacy_key_is_off_by_default(client, monkeypatch):
    monkeypatch.delenv("AUTH_ALLOW_API_KEY")
    assert client.get(PROTECTED, headers={"x-api-key": "legacy-test-key"}).status_code == 401


def test_legacy_key_accepted_when_explicitly_allowed(client):
    assert authorised(client.get(PROTECTED, headers={"x-api-key": "legacy-test-key"}))


def test_wrong_legacy_key_is_rejected(client):
    assert client.get(PROTECTED, headers={"x-api-key": "nope"}).status_code == 401


def test_legacy_key_can_be_switched_off(client, monkeypatch):
    monkeypatch.setenv("AUTH_ALLOW_API_KEY", "false")
    assert client.get(PROTECTED, headers={"x-api-key": "legacy-test-key"}).status_code == 401


def test_valid_token_still_works_with_legacy_key_switched_off(client, private_key, monkeypatch):
    monkeypatch.setenv("AUTH_ALLOW_API_KEY", "false")
    assert authorised(client.get(PROTECTED, headers=bearer(make_token(private_key))))


def test_bad_token_falls_back_to_valid_key_only_while_legacy_is_on(client, monkeypatch):
    headers = {"Authorization": "Bearer not.a.jwt", "x-api-key": "legacy-test-key"}
    assert authorised(client.get(PROTECTED, headers=headers))
    monkeypatch.setenv("AUTH_ALLOW_API_KEY", "false")
    assert client.get(PROTECTED, headers=headers).status_code == 401


def test_missing_api_key_env_never_matches(client, monkeypatch):
    monkeypatch.delenv("AI_API_KEY")
    assert client.get(PROTECTED, headers={"x-api-key": ""}).status_code == 401
    assert client.get(PROTECTED, headers={"x-api-key": "None"}).status_code == 401
