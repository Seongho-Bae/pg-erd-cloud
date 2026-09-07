from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, cast

import httpx
from fastapi import Depends, HTTPException, Request
from jose import jwt
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import ApiKey, UserAccount
from app.settings import settings


def _parse_oidc_algorithms(raw: str) -> list[str]:
    """Parse OIDC_ALGORITHMS into a non-empty allowlist.

    Security note:
    - JWT verification must *not* trust the token header's `alg`.
    - We pass an explicit allowlist to the verifier.
    """

    # Normalize and deduplicate so env values like "rs256, RS256" behave
    # predictably.
    normalized: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        alg = part.strip().upper()
        if not alg:
            continue
        # Defensive: never allow unsigned tokens.
        if alg == "NONE":
            continue
        # Defensive: never allow symmetric algorithms to prevent public key HMAC forgery.
        if alg.startswith("HS"):
            continue
        if alg in seen:
            continue
        seen.add(alg)
        normalized.append(alg)

    return normalized or ["RS256"]


@dataclass(frozen=True)
class CurrentUser:
    """Authenticated user identity used by API handlers."""

    user_account_uuid: uuid.UUID
    subject: str
    display_name: str | None


@dataclass(frozen=True)
class VerifiedToken:
    """Verified OIDC token details needed by auth and logout flows."""

    subject: str
    display_name: str | None
    jwt_id: str
    expires_at: dt.datetime


_oidc_config: dict[str, Any] | None = None
_oidc_jwks: dict[str, Any] | None = None
_oidc_config_expires_at: dt.datetime = dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
_oidc_jwks_expires_at: dt.datetime = dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
OIDC_ALLOWED_ALGORITHMS = tuple(_parse_oidc_algorithms(settings.oidc_algorithms))
OIDC_CONFIG_CACHE_TTL = dt.timedelta(minutes=10)
OIDC_JWKS_CACHE_TTL = dt.timedelta(minutes=5)
OIDC_JWKS_MIN_REFRESH_INTERVAL = dt.timedelta(seconds=60)
_last_jwks_refresh_at: dt.datetime = dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
_jwks_lock = asyncio.Lock()
OIDC_JWT_LEEWAY_SECONDS = 60
OIDC_ALLOWED_TOKEN_TYPES = {"jwt", "at+jwt"}


async def _get_oidc_config() -> dict:
    """Fetch and cache the OIDC discovery document."""
    if not settings.oidc_issuer:
        raise RuntimeError("OIDC is disabled")

    global _oidc_config, _oidc_config_expires_at
    now = dt.datetime.now(dt.timezone.utc)
    if _oidc_config is not None and now < _oidc_config_expires_at:
        return cast(dict, _oidc_config)

    async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
        r = await client.get(
            f"{settings.oidc_issuer.rstrip('/')}/.well-known/openid-configuration"
        )
        if r.is_redirect:
            raise RuntimeError("OIDC configuration endpoint must not redirect")
        r.raise_for_status()
        config = cast(dict[str, Any], r.json())

    _oidc_config = config
    _oidc_config_expires_at = now + OIDC_CONFIG_CACHE_TTL
    return cast(dict, config)


async def _get_jwks(force_refresh: bool = False) -> dict:
    """Fetch and cache the OIDC JWKS (signing keys)."""
    config = await _get_oidc_config()
    jwks_uri = config.get("jwks_uri")
    if not isinstance(jwks_uri, str):
        raise RuntimeError("OIDC jwks_uri missing")

    now = dt.datetime.now(dt.timezone.utc)

    if _oidc_jwks is not None:
        if not force_refresh and now < _oidc_jwks_expires_at:
            return cast(dict, _oidc_jwks)
        if (
            force_refresh
            and now < _last_jwks_refresh_at + OIDC_JWKS_MIN_REFRESH_INTERVAL
        ):
            return cast(dict, _oidc_jwks)

    async with _jwks_lock:
        now = dt.datetime.now(dt.timezone.utc)
        if _oidc_jwks is not None:
            if not force_refresh and now < _oidc_jwks_expires_at:
                return cast(dict, _oidc_jwks)
            if (
                force_refresh
                and now < _last_jwks_refresh_at + OIDC_JWKS_MIN_REFRESH_INTERVAL
            ):
                return cast(dict, _oidc_jwks)

        async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
            r = await client.get(jwks_uri)
            if r.is_redirect:
                raise RuntimeError("OIDC JWKS endpoint must not redirect")
            r.raise_for_status()
            jwks = cast(dict[str, Any], r.json())

        refreshed_at = dt.datetime.now(dt.timezone.utc)
        globals()["_oidc_jwks"] = jwks
        globals()["_oidc_jwks_expires_at"] = refreshed_at + OIDC_JWKS_CACHE_TTL
        globals()["_last_jwks_refresh_at"] = refreshed_at
        return cast(dict, jwks)


def _pick_jwk(jwks: dict, kid: str | None) -> dict | None:
    """Pick a JWK from a JWKS set by kid (or first if kid is None)."""
    keys = jwks.get("keys")
    if not isinstance(keys, list):
        return None
    for k in keys:
        if not isinstance(k, dict):
            continue
        if kid is None or k.get("kid") == kid:
            return k
    return None


def _jwt_expiry(claims: dict[str, Any]) -> dt.datetime:
    """Return the JWT expiry as an aware UTC datetime."""

    exp = claims.get("exp")
    if not isinstance(exp, int | float):
        raise HTTPException(status_code=401, detail="invalid token")
    return dt.datetime.fromtimestamp(float(exp), tz=dt.timezone.utc)


def _validate_jwt_header(header: dict[str, Any]) -> str:
    """Validate JOSE header fields before signature verification."""

    token_type = header.get("typ")
    if token_type is not None:
        if (
            not isinstance(token_type, str)
            or token_type.strip().lower() not in OIDC_ALLOWED_TOKEN_TYPES
        ):
            raise HTTPException(status_code=401, detail="invalid token")

    content_type = header.get("cty")
    if content_type is not None:
        raise HTTPException(status_code=401, detail="invalid token")

    header_alg_raw = header.get("alg")
    if not isinstance(header_alg_raw, str) or not header_alg_raw:
        raise HTTPException(status_code=401, detail="invalid token")
    return header_alg_raw.upper()


async def revoke_token_jti(jwt_id: str, expires_at: dt.datetime) -> None:
    """Record a JWT ID as revoked until its natural expiry."""

    from app.models import RevokedToken
    from app.db import SessionLocal

    if not jwt_id:
        return

    current = dt.datetime.now(dt.timezone.utc)
    async with SessionLocal() as session:
        await session.execute(
            delete(RevokedToken).where(RevokedToken.expires_at <= current)
        )
        revoked = RevokedToken(jwt_id=jwt_id, expires_at=expires_at)
        session.add(revoked)
        await session.commit()


async def is_token_jti_revoked(jwt_id: str) -> bool:
    """Return whether the JWT ID is currently revoked."""

    from app.models import RevokedToken
    from app.db import SessionLocal

    current = dt.datetime.now(dt.timezone.utc)
    async with SessionLocal() as session:
        stmt = select(RevokedToken).where(
            RevokedToken.jwt_id == jwt_id, RevokedToken.expires_at > current
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None


def _bearer_token_from_request(request: Request) -> str:
    """Return the bearer token from a request or fail authentication."""

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    return auth.split(" ", 1)[1].strip()


async def _decode_verified_oidc_token(token: str) -> dict[str, Any]:
    """Validate the JOSE header, signing key, and claims signature."""

    try:
        header = cast(dict[str, Any], jwt.get_unverified_header(token))
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="invalid token")

    header_alg = _validate_jwt_header(header)
    if header_alg not in OIDC_ALLOWED_ALGORITHMS:
        raise HTTPException(
            status_code=401,
            detail="invalid token",
        )

    jwks = await _get_jwks()
    jwk = _pick_jwk(jwks, header.get("kid"))
    if jwk is None:
        jwks = await _get_jwks(force_refresh=True)
        jwk = _pick_jwk(jwks, header.get("kid"))
    if jwk is None:
        raise HTTPException(status_code=401, detail="invalid token")

    kty = jwk.get("kty")
    if not isinstance(kty, str):
        raise HTTPException(status_code=401, detail="invalid token")
    jwk_kty = kty.upper()
    if jwk_kty == "RSA":
        if not (header_alg.startswith("RS") or header_alg.startswith("PS")):
            raise HTTPException(status_code=401, detail="invalid token")
    elif jwk_kty == "EC":
        if not header_alg.startswith("ES"):
            raise HTTPException(status_code=401, detail="invalid token")
    else:
        raise HTTPException(status_code=401, detail="invalid token")

    try:
        claims = jwt.decode(
            token,
            jwk,
            algorithms=list(OIDC_ALLOWED_ALGORITHMS),
            audience=settings.oidc_audience,
            issuer=settings.oidc_issuer,
            options={
                "verify_aud": bool(settings.oidc_audience),
                "require_aud": bool(settings.oidc_audience),
                "require_iss": True,
                "require_exp": True,
                "require_jti": True,
                "leeway": OIDC_JWT_LEEWAY_SECONDS,
            },
        )
    except Exception as err:
        raise HTTPException(
            status_code=401, detail="invalid token"
        ) from err

    return cast(dict[str, Any], claims)


async def _verified_token_from_claims(
    claims: dict[str, Any], verify_revocation: bool = True
) -> VerifiedToken:
    """Validate decoded claims and return the request auth identity."""

    sub = claims.get("sub")
    jwt_id = claims.get("jti")
    name = claims.get("name") or claims.get("preferred_username")
    if not isinstance(sub, str):
        raise HTTPException(status_code=401, detail="invalid token")
    if not isinstance(jwt_id, str) or not jwt_id.strip():
        raise HTTPException(status_code=401, detail="invalid token")

    expires_at = _jwt_expiry(claims)
    if verify_revocation and await is_token_jti_revoked(jwt_id):
        raise HTTPException(status_code=401, detail="invalid token")

    return VerifiedToken(
        subject=sub,
        display_name=str(name) if isinstance(name, str) else None,
        jwt_id=jwt_id,
        expires_at=expires_at,
    )


async def _get_verified_token_from_request(
    request: Request, verify_revocation: bool = True
) -> VerifiedToken:
    """Extract and verify OIDC token claims from a request."""

    if settings.oidc_issuer:
        token = _bearer_token_from_request(request)
        claims = await _decode_verified_oidc_token(token)
        return await _verified_token_from_claims(claims, verify_revocation)

    raise HTTPException(status_code=500, detail="OIDC configuration required")


async def _get_subject_from_request(
    request: Request, verify_revocation: bool = True
) -> tuple[str, str | None]:
    """Extract (subject, display_name) from a verified request token."""

    verified = await _get_verified_token_from_request(request, verify_revocation)
    return verified.subject, verified.display_name


async def try_get_subject_for_rate_limit(request: Request) -> str | None:
    """Best-effort subject extraction for rate limiting.

    This helper is intentionally lightweight:
    - It must NOT touch the DB (unlike get_current_user).
    - It must NOT change auth behavior. Missing/invalid auth returns None so
      unauthenticated requests can still be limited by IP.
    """

    try:
        subject, _ = await _get_subject_from_request(request, verify_revocation=False)
        return subject
    except HTTPException:
        return None


_user_cache: dict[str, tuple[CurrentUser, dt.datetime]] = {}
USER_CACHE_MAX_SIZE = 1000
USER_CACHE_TTL = dt.timedelta(minutes=5)


async def _ensure_user(
    session: AsyncSession, subject: str, display_name: str | None
) -> CurrentUser:
    """Get or create a UserAccount for the given OIDC subject."""
    now = dt.datetime.now(dt.timezone.utc)
    cached = _user_cache.get(subject)
    if cached is not None:
        user, expires_at = cached
        if now < expires_at:
            return user
        else:
            del _user_cache[subject]

    row = await session.execute(
        select(UserAccount).where(UserAccount.oidc_subject == subject)
    )
    existing = row.scalars().first()
    if existing is not None:
        user = CurrentUser(
            user_account_uuid=existing.user_account_uuid,
            subject=existing.oidc_subject,
            display_name=existing.display_name,
        )
    else:
        user_account = UserAccount(
            user_account_uuid=uuid.uuid4(),
            oidc_subject=subject,
            display_name=display_name,
            created_at=now,
        )
        session.add(user_account)
        await session.flush()
        user = CurrentUser(
            user_account_uuid=user_account.user_account_uuid,
            subject=user_account.oidc_subject,
            display_name=user_account.display_name,
        )

    if len(_user_cache) >= USER_CACHE_MAX_SIZE:
        _user_cache.clear()

    _user_cache[subject] = (user, now + USER_CACHE_TTL)
    return user


API_KEY_PREFIX = "pgerd_"
API_KEY_PBKDF2_ITERATIONS = 210_000


def hash_api_key(token: str) -> str:
    """Deterministic PBKDF2-HMAC digest of an API key (indexable lookup).

    API keys are server-generated random bearer credentials, but CodeQL treats
    bearer tokens as sensitive password-like inputs. PBKDF2-HMAC-SHA256 keeps
    storage deterministic for the unique-indexed lookup while avoiding a fast
    raw SHA-256/HMAC digest if the API-key table is disclosed. ``app_secret`` is
    used as a deployment pepper, so database-only disclosure is not enough to
    test guessed keys offline.
    """

    return hashlib.pbkdf2_hmac(
        "sha256",
        token.encode("utf-8"),
        settings.app_secret.encode("utf-8"),
        API_KEY_PBKDF2_ITERATIONS,
    ).hex()


async def _user_from_api_key(session: AsyncSession, token: str) -> CurrentUser:
    """Authenticate an ``Authorization: Bearer pgerd_...`` API key.

    Looks up the PBKDF2 hash (constant-shape errors: any failure is the same
    401 so keys cannot be probed) and rejects revoked keys.
    """

    row = await session.execute(
        select(ApiKey, UserAccount)
        .join(UserAccount, UserAccount.user_account_uuid == ApiKey.user_account_uuid)
        .where(ApiKey.key_hash == hash_api_key(token))
    )
    pair = row.first()
    if pair is None or pair[0].revoked_at is not None:
        raise HTTPException(status_code=401, detail="invalid token")
    user = pair[1]
    return CurrentUser(
        user_account_uuid=user.user_account_uuid,
        subject=user.oidc_subject,
        display_name=user.display_name,
    )


async def get_current_user(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> CurrentUser:
    """FastAPI dependency that authenticates and returns the current user.

    Accepts either the standard OIDC bearer token or a ``pgerd_``-prefixed
    API key (programmatic access for CI/CD and SDKs).
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer " + API_KEY_PREFIX):
        return await _user_from_api_key(session, auth_header[len("Bearer "):])
    subject, display_name = await _get_subject_from_request(request)
    async with session.begin():
        return await _ensure_user(session, subject, display_name)


async def revoke_current_request_token(request: Request) -> None:
    """Revoke the current request token until its natural expiry."""

    verified = await _get_verified_token_from_request(request)
    await revoke_token_jti(verified.jwt_id, verified.expires_at)
