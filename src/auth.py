"""Request authentication for the API.

Preferred: a short-lived OIDC token that Vercel issues to the website's
functions (``Authorization: Bearer <jwt>``), verified against Vercel's public
keys. There is no shared secret to store, leak or rotate.

The legacy static key (``x-api-key``) is off by default. It is only accepted
when ``AUTH_ALLOW_API_KEY`` is set to ``true`` and ``AI_API_KEY`` is set.

Configuration (environment variables):
    OIDC_ISSUER             e.g. https://oidc.vercel.com/<team-slug>
    OIDC_AUDIENCE           e.g. https://vercel.com/<team-slug>
    OIDC_ALLOWED_SUBJECTS   comma-separated exact ``sub`` values, e.g.
                            owner:<team>:project:<project>:environment:production
    AI_API_KEY              legacy static key (only used when allowed below)
    AUTH_ALLOW_API_KEY      "true" re-enables the legacy key (default: off)

OIDC is only enabled when issuer, audience and at least one allowed subject
are all set, so a half-finished configuration never lets anyone in.
"""

import hmac
import logging
import os
from functools import wraps

import jwt
from flask import g, jsonify, request

logger = logging.getLogger(__name__)

JWKS_CACHE_SECONDS = 3600
JWKS_TIMEOUT_SECONDS = 5
CLOCK_SKEW_SECONDS = 30

# Only asymmetric signatures are accepted. Never allow "none" or HS* here,
# otherwise a token signed with the public key as an HMAC secret would pass.
ALLOWED_ALGORITHMS = ["RS256"]
REQUIRED_CLAIMS = ["exp", "iat", "iss", "aud", "sub"]

_jwks_clients = {}


class AuthError(Exception):
    """Raised when a credential is missing, malformed or not allowed."""


def _csv(name):
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


def _get_jwks_client(issuer):
    client = _jwks_clients.get(issuer)
    if client is None:
        client = jwt.PyJWKClient(
            f"{issuer.rstrip('/')}/.well-known/jwks",
            cache_jwk_set=True,
            lifespan=JWKS_CACHE_SECONDS,
            timeout=JWKS_TIMEOUT_SECONDS,
        )
        _jwks_clients[issuer] = client
    return client


def verify_oidc_token(token):
    """Return the verified claims of ``token`` or raise ``AuthError``."""
    issuer = os.getenv("OIDC_ISSUER", "").strip()
    audience = os.getenv("OIDC_AUDIENCE", "").strip()
    subjects = _csv("OIDC_ALLOWED_SUBJECTS")
    if not issuer or not audience or not subjects:
        raise AuthError("OIDC is not configured")

    try:
        signing_key = _get_jwks_client(issuer).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=ALLOWED_ALGORITHMS,
            audience=audience,
            issuer=issuer,
            leeway=CLOCK_SKEW_SECONDS,
            options={"require": REQUIRED_CLAIMS},
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"{type(exc).__name__}: {exc}") from exc

    if claims.get("sub") not in subjects:
        raise AuthError("subject is not allowed")
    return claims


def _bearer_token():
    scheme, _, value = request.headers.get("Authorization", "").partition(" ")
    value = value.strip()
    if scheme.lower() == "bearer" and value:
        return value
    return None


def _legacy_key_valid():
    if os.getenv("AUTH_ALLOW_API_KEY", "false").strip().lower() != "true":
        return False
    expected = os.getenv("AI_API_KEY")
    provided = request.headers.get("x-api-key")
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def require_auth(func):
    """Allow the request only with a valid OIDC token or (transition) API key.

    On success the verified caller is stored in ``g.auth_identity`` (used as the
    per-caller rate-limit key).
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        reason = "no credentials"

        token = _bearer_token()
        if token:
            try:
                claims = verify_oidc_token(token)
            except AuthError as exc:
                reason = f"oidc token rejected ({exc})"
            else:
                g.auth_identity = f"oidc:{claims['sub']}"
                return func(*args, **kwargs)

        if _legacy_key_valid():
            g.auth_identity = "api-key"
            return func(*args, **kwargs)

        logger.warning("Unauthorized request to %s: %s", request.path, reason)
        response = jsonify({"error": "Unauthorized"})
        response.status_code = 401
        response.headers["WWW-Authenticate"] = "Bearer"
        return response

    return wrapper
