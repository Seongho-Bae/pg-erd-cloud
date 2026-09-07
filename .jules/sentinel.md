## 2025-02-18 - Hardening Pydantic String Fields Against Control Characters
**Vulnerability:** User-provided string fields (like project and connection names) lacked strict validation against control characters, only relying on length constraints.
**Learning:** This could potentially lead to Log Injection (CRLF injection), Null Byte Injection, or terminal escape injection if these strings are subsequently logged or rendered directly.
**Prevention:** Use explicit regex validation `pattern=r'^[^\x00-\x1F\x7F]+$'` on Pydantic string fields to strictly reject control characters.
## 2026-09-06 - Generic Authentication Error Messages
**Vulnerability:** JWT validation errors returned highly specific details (e.g., "token missing exp", "algorithm/key type mismatch").
**Learning:** This specific information could be leveraged by attackers for information gathering or side-channel attacks during authentication.
**Prevention:** Ensure all authentication-related exceptions, particularly JWT validation errors, throw a generic 401 response such as `detail="invalid token"` without exposing internal validation details.
