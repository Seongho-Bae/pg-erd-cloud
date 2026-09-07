from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app import auth
from app.settings import settings


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RS256", ["RS256"]),
        ("rs256, ES256", ["RS256", "ES256"]),
        ("RS256, rs256", ["RS256"]),
        ("none, RS256", ["RS256"]),
        ("none", ["RS256"]),
        (", , ", ["RS256"]),
        ("HS256, RS256", ["RS256"]),
        ("RS256, hs256", ["RS256"]),
        ("HS256, HS384, HS512", ["RS256"]),
    ],
)
def test_parse_oidc_algorithms(raw: str, expected: list[str]) -> None:
    assert auth._parse_oidc_algorithms(raw) == expected


def make_request(headers: dict[str, str] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/me",
            "headers": [
                (name.lower().encode("latin1"), value.encode("latin1"))
                for name, value in (headers or {}).items()
            ],
        }
    )


def exp_claim() -> int:
    return int(auth.dt.datetime.now(auth.dt.timezone.utc).timestamp()) + 300


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, object], is_redirect: bool = False) -> None:
        self._payload = payload
        self.is_redirect = is_redirect
        self.raise_for_status_called = False

    def raise_for_status(self) -> None:
        self.raise_for_status_called = True

    def json(self) -> dict[str, object]:
        return self._payload


@pytest.mark.asyncio
async def test_oidc_config_fetch_disables_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeHttpResponse({"jwks_uri": "https://issuer.example/jwks"})
    observed: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get(self, url: str) -> _FakeHttpResponse:
            observed["url"] = url
            return response

    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example/")
    monkeypatch.setattr(auth.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(auth, "_oidc_config", None)
    monkeypatch.setattr(
        auth,
        "_oidc_config_expires_at",
        auth.dt.datetime.fromtimestamp(0, tz=auth.dt.timezone.utc),
    )

    config = await auth._get_oidc_config()

    assert config == {"jwks_uri": "https://issuer.example/jwks"}
    assert observed["timeout"] == 5
    assert observed["follow_redirects"] is False
    assert observed["url"] == "https://issuer.example/.well-known/openid-configuration"
    assert response.raise_for_status_called is True


@pytest.mark.asyncio
async def test_oidc_config_rejects_redirect_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeHttpResponse({"jwks_uri": "https://issuer.example/jwks"}, True)

    class FakeAsyncClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get(self, _url: str) -> _FakeHttpResponse:
            return response

    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example")
    monkeypatch.setattr(auth.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(auth, "_oidc_config", None)
    monkeypatch.setattr(
        auth,
        "_oidc_config_expires_at",
        auth.dt.datetime.fromtimestamp(0, tz=auth.dt.timezone.utc),
    )

    with pytest.raises(RuntimeError, match="must not redirect"):
        await auth._get_oidc_config()

    assert response.raise_for_status_called is False


@pytest.mark.asyncio
async def test_jwks_fetch_disables_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeHttpResponse({"keys": []})
    observed: dict[str, object] = {}

    async def fake_config() -> dict[str, object]:
        return {"jwks_uri": "https://issuer.example/jwks"}

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get(self, url: str) -> _FakeHttpResponse:
            observed["url"] = url
            return response

    monkeypatch.setattr(auth, "_get_oidc_config", fake_config)
    monkeypatch.setattr(auth.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(auth, "_oidc_jwks", None)
    monkeypatch.setattr(
        auth,
        "_oidc_jwks_expires_at",
        auth.dt.datetime.fromtimestamp(0, tz=auth.dt.timezone.utc),
    )

    jwks = await auth._get_jwks()

    assert jwks == {"keys": []}
    assert observed["timeout"] == 5
    assert observed["follow_redirects"] is False
    assert observed["url"] == "https://issuer.example/jwks"
    assert response.raise_for_status_called is True


@pytest.mark.asyncio
async def test_oidc_rejects_header_selected_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "OIDC_ALLOWED_ALGORITHMS", ("RS256",))
    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example")
    monkeypatch.setattr(settings, "oidc_audience", "pg-erd")
    monkeypatch.setattr(
        auth.jwt, "get_unverified_header", lambda _: {"kid": "key-1", "alg": "HS256"}
    )

    async def fake_jwks() -> dict:
        return {"keys": [{"kid": "key-1", "kty": "RSA"}]}

    monkeypatch.setattr(auth, "_get_jwks", fake_jwks)

    async def mock_is_token_revoked2(jti):
        return jti == "revoked-jwt"

    monkeypatch.setattr(auth, "is_token_jti_revoked", mock_is_token_revoked2)

    def fail_decode(*_: object, **__: object) -> dict:
        raise AssertionError("jwt.decode must not run for unsupported algorithms")

    monkeypatch.setattr(auth.jwt, "decode", fail_decode)

    with pytest.raises(HTTPException) as exc_info:
        await auth._get_subject_from_request(
            make_request({"Authorization": "Bearer token"})
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "invalid token"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "jwk",
    [
        {"kid": "key-1", "kty": "EC"},
        {"kid": "key-1", "kty": "oct"},
        {"kid": "key-1"},
    ],
)
async def test_oidc_decode_rejects_kty_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    jwk: dict[str, object],
) -> None:
    monkeypatch.setattr(auth, "OIDC_ALLOWED_ALGORITHMS", ("RS256",))
    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example")
    monkeypatch.setattr(settings, "oidc_audience", "pg-erd")
    monkeypatch.setattr(
        auth.jwt, "get_unverified_header", lambda _: {"kid": "key-1", "alg": "RS256"}
    )

    async def fake_jwks() -> dict:
        return {"keys": [jwk]}

    monkeypatch.setattr(auth, "_get_jwks", fake_jwks)

    def fail_decode(*_: object, **__: object) -> dict:
        raise AssertionError("jwt.decode must not run for mismatched key types")

    monkeypatch.setattr(auth.jwt, "decode", fail_decode)

    with pytest.raises(HTTPException) as exc_info:
        await auth._decode_verified_oidc_token("ey...fake...")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "invalid token"


@pytest.mark.asyncio
async def test_oidc_decode_uses_fixed_algorithm_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "OIDC_ALLOWED_ALGORITHMS", ("RS256",))
    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example")
    monkeypatch.setattr(settings, "oidc_audience", "pg-erd")
    monkeypatch.setattr(
        auth.jwt, "get_unverified_header", lambda _: {"kid": "key-1", "alg": "RS256"}
    )

    async def fake_jwks() -> dict:
        return {"keys": [{"kid": "key-1", "kty": "RSA"}]}

    observed: dict[str, object] = {}

    def fake_decode(*args: object, **kwargs: object) -> dict:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"sub": "user-1", "name": "User One", "jti": "jwt-1", "exp": exp_claim()}

    monkeypatch.setattr(auth, "_get_jwks", fake_jwks)

    async def mock_is_token_revoked2(jti):
        return jti == "revoked-jwt"

    monkeypatch.setattr(auth, "is_token_jti_revoked", mock_is_token_revoked2)
    monkeypatch.setattr(auth.jwt, "decode", fake_decode)

    async def mock_is_token_revoked(jti):
        return jti == "revoked-jwt"

    monkeypatch.setattr(auth, "is_token_jti_revoked", mock_is_token_revoked)

    subject, display_name = await auth._get_subject_from_request(
        make_request({"Authorization": "Bearer token"})
    )

    assert subject == "user-1"
    assert display_name == "User One"
    assert observed["kwargs"] == {
        "algorithms": ["RS256"],
        "audience": "pg-erd",
        "issuer": "https://issuer.example",
        "options": {
            "verify_aud": True,
            "require_aud": True,
            "require_iss": True,
            "require_exp": True,
            "require_jti": True,
            "leeway": auth.OIDC_JWT_LEEWAY_SECONDS,
        },
    }


@pytest.mark.parametrize(
    ("header", "detail"),
    [
        (
            {"kid": "key-1", "alg": "RS256", "typ": "nested+jwt"},
            "invalid token",
        ),
        (
            {"kid": "key-1", "alg": "RS256", "cty": "JWT"},
            "invalid token",
        ),
    ],
)
@pytest.mark.asyncio
async def test_oidc_rejects_unsupported_header_types(
    monkeypatch: pytest.MonkeyPatch, header: dict[str, str], detail: str
) -> None:
    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example")
    monkeypatch.setattr(settings, "oidc_audience", "pg-erd")
    monkeypatch.setattr(auth.jwt, "get_unverified_header", lambda _: header)

    async def fail_jwks() -> dict:
        raise AssertionError("JWKS must not load for unsupported token headers")

    monkeypatch.setattr(auth, "_get_jwks", fail_jwks)

    with pytest.raises(HTTPException) as exc_info:
        await auth._get_subject_from_request(
            make_request({"Authorization": "Bearer token"})
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == detail


@pytest.mark.asyncio
async def test_oidc_refreshes_jwks_when_kid_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "OIDC_ALLOWED_ALGORITHMS", ("RS256",))
    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example")
    monkeypatch.setattr(settings, "oidc_audience", "pg-erd")
    monkeypatch.setattr(
        auth.jwt, "get_unverified_header", lambda _: {"kid": "new-key", "alg": "RS256"}
    )

    refresh_calls: list[bool] = []

    async def fake_jwks(force_refresh: bool = False) -> dict:
        refresh_calls.append(force_refresh)
        if force_refresh:
            return {"keys": [{"kid": "new-key", "kty": "RSA"}]}
        return {"keys": [{"kid": "old-key", "kty": "RSA"}]}

    observed: dict[str, object] = {}

    def fake_decode(*args: object, **kwargs: object) -> dict:
        observed["key"] = args[1]
        observed["kwargs"] = kwargs
        return {"sub": "user-1", "name": "User One", "jti": "jwt-1", "exp": exp_claim()}

    monkeypatch.setattr(auth, "_get_jwks", fake_jwks)

    async def mock_is_token_revoked2(jti):
        return jti == "revoked-jwt"

    monkeypatch.setattr(auth, "is_token_jti_revoked", mock_is_token_revoked2)
    monkeypatch.setattr(auth.jwt, "decode", fake_decode)

    async def mock_is_token_revoked(jti):
        return jti == "revoked-jwt"

    monkeypatch.setattr(auth, "is_token_jti_revoked", mock_is_token_revoked)

    subject, display_name = await auth._get_subject_from_request(
        make_request({"Authorization": "Bearer token"})
    )

    assert subject == "user-1"
    assert display_name == "User One"
    assert refresh_calls == [False, True]
    assert observed["key"] == {"kid": "new-key", "kty": "RSA"}


@pytest.mark.asyncio
async def test_oidc_requires_jti_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example")
    monkeypatch.setattr(settings, "oidc_audience", "pg-erd")
    monkeypatch.setattr(
        auth.jwt, "get_unverified_header", lambda _: {"kid": "key-1", "alg": "RS256"}
    )

    async def fake_jwks() -> dict:
        return {"keys": [{"kid": "key-1", "kty": "RSA"}]}

    monkeypatch.setattr(auth, "_get_jwks", fake_jwks)

    async def mock_is_token_revoked2(jti):
        return jti == "revoked-jwt"

    monkeypatch.setattr(auth, "is_token_jti_revoked", mock_is_token_revoked2)
    monkeypatch.setattr(
        auth.jwt,
        "decode",
        lambda *_args, **_kwargs: {"sub": "user-1", "exp": exp_claim()},
    )

    with pytest.raises(HTTPException) as exc_info:
        await auth._get_subject_from_request(
            make_request({"Authorization": "Bearer token"})
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "invalid token"


@pytest.mark.asyncio
async def test_oidc_rejects_revoked_jti(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example")
    monkeypatch.setattr(settings, "oidc_audience", "pg-erd")
    monkeypatch.setattr(
        auth.jwt, "get_unverified_header", lambda _: {"kid": "key-1", "alg": "RS256"}
    )

    async def fake_jwks() -> dict:
        return {"keys": [{"kid": "key-1", "kty": "RSA"}]}

    expires_at = auth.dt.datetime.now(auth.dt.timezone.utc) + auth.dt.timedelta(
        minutes=5
    )
    monkeypatch.setattr(auth, "_get_jwks", fake_jwks)

    async def mock_is_token_revoked2(jti):
        return jti == "revoked-jwt"

    monkeypatch.setattr(auth, "is_token_jti_revoked", mock_is_token_revoked2)
    monkeypatch.setattr(
        auth.jwt,
        "decode",
        lambda *_args, **_kwargs: {
            "sub": "user-1",
            "jti": "revoked-jwt",
            "exp": int(expires_at.timestamp()),
        },
    )

    async def mock_revoke(jti, ext):
        pass

    monkeypatch.setattr(auth, "revoke_token_jti", mock_revoke)
    await auth.revoke_token_jti("revoked-jwt", expires_at)

    with pytest.raises(HTTPException) as exc_info:
        await auth._get_subject_from_request(
            make_request({"Authorization": "Bearer token"})
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "invalid token"


@pytest.mark.asyncio
async def test_auth_fails_closed_without_oidc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "oidc_issuer", None)

    with pytest.raises(HTTPException) as exc_info:
        await auth._get_subject_from_request(make_request({"X-Dev-User": "local"}))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "OIDC configuration required"


class _FakeScalarResult:
    def __init__(self, user: object | None) -> None:
        self._user = user

    def first(self) -> object | None:
        return self._user


class _FakeExecuteResult:
    def __init__(self, user: object | None) -> None:
        self._user = user

    def scalars(self) -> _FakeScalarResult:
        return _FakeScalarResult(self._user)


class _FakeSession:
    def __init__(self, user: object | None) -> None:
        self._user = user
        self.execute_calls = 0
        self.added: list[object] = []
        self.flush_calls = 0

    async def execute(self, _statement: object) -> _FakeExecuteResult:
        self.execute_calls += 1
        return _FakeExecuteResult(self._user)

    def add(self, value: object) -> None:
        self.added.append(value)

    async def flush(self) -> None:
        self.flush_calls += 1


class _ExistingUser:
    def __init__(self) -> None:
        self.user_account_uuid = uuid.uuid4()
        self.oidc_subject = "subject-1"
        self.display_name = "User One"


@pytest.mark.asyncio
async def test_ensure_user_reuses_short_lived_cache() -> None:
    auth._user_cache.clear()
    try:
        session = _FakeSession(_ExistingUser())

        first = await auth._ensure_user(session, "subject-1", "User One")
        second = await auth._ensure_user(session, "subject-1", "Changed")

        assert first == second
        assert session.execute_calls == 1
        assert session.added == []
        assert session.flush_calls == 0
    finally:
        auth._user_cache.clear()


@pytest.mark.asyncio
async def test_try_get_subject_for_rate_limit_error_path():
    """Verify try_get_subject_for_rate_limit returns None on auth failure."""
    req = make_request()  # No Authorization header

    # We should get None because of the Missing Bearer Token HTTPException
    subject = await auth.try_get_subject_for_rate_limit(req)
    assert subject is None


async def test_oidc_decode_rejects_invalid_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mock_get_unverified_header(token):
        raise Exception("Invalid header")

    monkeypatch.setattr(auth.jwt, "get_unverified_header", mock_get_unverified_header)

    with pytest.raises(HTTPException) as excinfo:
        await auth._decode_verified_oidc_token("invalid_token")

    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "invalid token"


@pytest.mark.asyncio
async def test_oidc_decode_rejects_jwt_decode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example")
    monkeypatch.setattr(settings, "oidc_audience", "pg-erd")
    monkeypatch.setattr(auth, "OIDC_ALLOWED_ALGORITHMS", ("RS256",))
    monkeypatch.setattr(
        auth.jwt, "get_unverified_header", lambda _: {"kid": "key-1", "alg": "RS256"}
    )

    async def fake_jwks() -> dict:
        return {"keys": [{"kid": "key-1", "kty": "RSA"}]}

    def fail_decode(*_args: object, **_kwargs: object) -> dict:
        raise auth.jwt.PyJWTError("mocked decoding error")

    monkeypatch.setattr(auth, "_get_jwks", fake_jwks)

    async def mock_is_token_revoked2(jti):
        return jti == "revoked-jwt"

    monkeypatch.setattr(auth, "is_token_jti_revoked", mock_is_token_revoked2)
    monkeypatch.setattr(auth.jwt, "decode", fail_decode)

    with pytest.raises(HTTPException) as exc_info:
        await auth._decode_verified_oidc_token("Bearer token")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "invalid token"


@pytest.mark.asyncio
async def test_oidc_rejects_algorithm_key_type_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "OIDC_ALLOWED_ALGORITHMS", ("HS256", "RS256"))
    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example")
    monkeypatch.setattr(settings, "oidc_audience", "pg-erd")
    monkeypatch.setattr(
        auth.jwt, "get_unverified_header", lambda _: {"kid": "key-1", "alg": "HS256"}
    )

    async def fake_jwks() -> dict:
        # JWK says RSA, but header says HS256
        return {"keys": [{"kid": "key-1", "kty": "RSA"}]}

    monkeypatch.setattr(auth, "_get_jwks", fake_jwks)

    async def mock_is_token_revoked2(jti):
        return jti == "revoked-jwt"

    monkeypatch.setattr(auth, "is_token_jti_revoked", mock_is_token_revoked2)

    def fail_decode(*_: object, **__: object) -> dict:
        raise AssertionError(
            "jwt.decode must not run for mismatched algorithm/key type"
        )

    monkeypatch.setattr(auth.jwt, "decode", fail_decode)

    with pytest.raises(HTTPException) as exc_info:
        await auth._decode_verified_oidc_token("ey...")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "invalid token"


@pytest.mark.asyncio
async def test_oidc_jwks_refresh_rate_limiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_count = 0

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get(self, url: str) -> _FakeHttpResponse:
            nonlocal request_count
            request_count += 1
            if url.endswith("openid-configuration"):
                return _FakeHttpResponse({"jwks_uri": "https://issuer.example/jwks"})
            return _FakeHttpResponse({"keys": [{"kid": "new-key", "kty": "RSA"}]})

    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example")
    monkeypatch.setattr(auth.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(auth, "_oidc_config", None)
    monkeypatch.setattr(auth, "_oidc_jwks", None)
    monkeypatch.setattr(
        auth,
        "_oidc_jwks_expires_at",
        auth.dt.datetime.fromtimestamp(0, tz=auth.dt.timezone.utc),
    )
    monkeypatch.setattr(
        auth,
        "_last_jwks_refresh_at",
        auth.dt.datetime.fromtimestamp(0, tz=auth.dt.timezone.utc),
    )

    jwks = await auth._get_jwks()
    assert jwks == {"keys": [{"kid": "new-key", "kty": "RSA"}]}
    assert request_count == 2

    before_second_refresh = request_count
    jwks2 = await auth._get_jwks(force_refresh=True)
    assert jwks2 == {"keys": [{"kid": "new-key", "kty": "RSA"}]}
    assert request_count == before_second_refresh


@pytest.mark.asyncio
async def test_oidc_jwks_force_refresh_is_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_count = 0

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get(self, url: str) -> _FakeHttpResponse:
            nonlocal request_count
            request_count += 1
            if url.endswith("openid-configuration"):
                return _FakeHttpResponse({"jwks_uri": "https://issuer.example/jwks"})
            await asyncio.sleep(0)
            return _FakeHttpResponse({"keys": [{"kid": "new-key", "kty": "RSA"}]})

    monkeypatch.setattr(settings, "oidc_issuer", "https://issuer.example")
    monkeypatch.setattr(auth.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(auth, "_oidc_config", None)
    monkeypatch.setattr(auth, "_oidc_jwks", None)
    monkeypatch.setattr(
        auth,
        "_oidc_jwks_expires_at",
        auth.dt.datetime.fromtimestamp(0, tz=auth.dt.timezone.utc),
    )
    monkeypatch.setattr(
        auth,
        "_last_jwks_refresh_at",
        auth.dt.datetime.fromtimestamp(0, tz=auth.dt.timezone.utc),
    )

    await auth._get_jwks()
    before_concurrent_refresh = request_count
    monkeypatch.setattr(
        auth,
        "_last_jwks_refresh_at",
        auth.dt.datetime.fromtimestamp(0, tz=auth.dt.timezone.utc),
    )

    refreshed = await asyncio.gather(
        *(auth._get_jwks(force_refresh=True) for _ in range(5))
    )

    assert refreshed == [
        {"keys": [{"kid": "new-key", "kty": "RSA"}]},
        {"keys": [{"kid": "new-key", "kty": "RSA"}]},
        {"keys": [{"kid": "new-key", "kty": "RSA"}]},
        {"keys": [{"kid": "new-key", "kty": "RSA"}]},
        {"keys": [{"kid": "new-key", "kty": "RSA"}]},
    ]
    assert request_count == before_concurrent_refresh + 1
