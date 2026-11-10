import pytest

import auth
from app import app, limiter
from oidc import AUDIENCE, ISSUER, PREVIEW_SUBJECT, SUBJECT, FakeJwksClient, make_token

PROTECTED = "/get_random_word"  # 60 per hour per caller


@pytest.fixture
def client(monkeypatch, private_key):
    monkeypatch.setenv("OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("OIDC_ALLOWED_SUBJECTS", f"{SUBJECT},{PREVIEW_SUBJECT}")
    monkeypatch.setenv("AI_API_KEY", "legacy-test-key")
    monkeypatch.setenv("AUTH_ALLOW_API_KEY", "true")
    monkeypatch.setattr(
        auth, "_get_jwks_client", lambda _issuer: FakeJwksClient(private_key.public_key())
    )
    app.config["TESTING"] = True
    limiter.enabled = True
    limiter.reset()
    try:
        with app.test_client() as test_client:
            yield test_client
    finally:
        limiter.reset()
        limiter.enabled = False


def bearer(key, **claims):
    return {"Authorization": f"Bearer {make_token(key, **claims)}"}


def test_each_verified_caller_has_its_own_budget(client, private_key):
    """All traffic comes from the website's servers, yet callers do not share a limit."""
    production = bearer(private_key, sub=SUBJECT)
    preview = bearer(private_key, sub=PREVIEW_SUBJECT)

    for _ in range(60):
        assert client.get(PROTECTED, headers=production).status_code in (200, 503)
    assert client.get(PROTECTED, headers=production).status_code == 429
    assert client.get(PROTECTED, headers=preview).status_code in (200, 503)


def test_legacy_key_is_a_separate_caller(client, private_key):
    production = bearer(private_key, sub=SUBJECT)
    for _ in range(60):
        client.get(PROTECTED, headers=production)
    assert client.get(PROTECTED, headers=production).status_code == 429
    assert client.get(PROTECTED, headers={"x-api-key": "legacy-test-key"}).status_code in (200, 503)


def test_unauthenticated_requests_do_not_use_a_callers_budget(client, private_key):
    production = bearer(private_key, sub=SUBJECT)
    for _ in range(5):
        assert client.get(PROTECTED).status_code == 401
    for _ in range(60):
        assert client.get(PROTECTED, headers=production).status_code in (200, 503)


def test_ip_shield_throttles_unauthenticated_floods(client):
    statuses = {client.get(PROTECTED).status_code for _ in range(305)}
    assert statuses == {401, 429}
