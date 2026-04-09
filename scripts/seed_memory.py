"""Seed memory script for ChronoAgent.

Populates four persistent ChromaDB collections with structured knowledge:

* ``security_patterns``  — ~50 CWE-mapped vulnerability detection notes.
* ``style_conventions``  — ~30 code quality and style rules.
* ``report_templates``   — ~10 PR review report structure templates.
* ``sample_reviews``     — ~20 synthetic PR review summaries for RAG context.

Usage::

    # Seed to default ./chroma_data (from project root)
    py -m scripts.seed_memory

    # Custom directory
    py -m scripts.seed_memory --chroma-dir /path/to/chroma

    # Wipe existing collections and re-seed
    py -m scripts.seed_memory --reset

All collections use :class:`~chronoagent.agents.backends.mock.MockBackend` for
embeddings so no API keys are required.  For production use, swap the backend
with :class:`~chronoagent.agents.backends.together.TogetherAIBackend`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import chromadb
from chromadb.api import ClientAPI

from chronoagent.agents.backends.mock import MockBackend

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security vulnerability patterns (~50 CWE entries)
# First 25 are the CWE Top-25; next 25 cover additional high-impact classes.
# ---------------------------------------------------------------------------

SECURITY_PATTERNS: Final[list[str]] = [
    # --- CWE Top-25 (core set) -----------------------------------------------
    "CWE-89 SQL Injection: Always use parameterized queries or ORM abstractions. "
    "Never interpolate user input directly into SQL strings. "
    "Detection: string concatenation in DB query construction.",
    "CWE-79 Cross-site Scripting (XSS): Escape all user-supplied data before rendering "
    "in HTML. Use Content-Security-Policy headers to restrict script execution. "
    "Detection: unsanitized template variables, missing output encoding.",
    "CWE-287 Improper Authentication: Use strong password hashing (bcrypt, argon2). "
    "Enforce MFA on privileged accounts. Invalidate sessions on logout. "
    "Detection: plain-text password storage, missing session invalidation.",
    "CWE-798 Hardcoded Credentials: Never hardcode API keys, passwords, or tokens. "
    "Use environment variables or a secrets manager (Vault, AWS Secrets Manager). "
    "Detection: literals matching /password|api_key|secret/ in source code.",
    "CWE-20 Improper Input Validation: Validate all external inputs at the application "
    "boundary. Use allowlists, not denylists, for permitted values. "
    "Detection: missing schema validation on request bodies or query parameters.",
    "CWE-352 Cross-Site Request Forgery (CSRF): Require CSRF tokens on all "
    "state-modifying requests. Validate the Origin and Referer headers. "
    "Detection: POST/PUT/DELETE handlers missing token verification.",
    "CWE-22 Path Traversal: Normalize and validate all file paths. Restrict uploads "
    "to a designated safe directory; reject any '../' sequences. "
    "Detection: user-controlled path components without normalization.",
    "CWE-937 Using Components with Known Vulnerabilities: Scan dependencies for known "
    "CVEs on every build. Pin transitive dependencies; review changelogs before upgrading. "
    "Detection: outdated package versions in lock files.",
    "CWE-347 Improper Verification of Cryptographic Signature (JWT): Always validate "
    "the algorithm field; reject 'none'. Use RS256 or ES256 for asymmetric signing. "
    "Detection: algorithm field not explicitly allowlisted in JWT validation.",
    "CWE-307 Improper Restriction of Excessive Authentication Attempts: Apply rate "
    "limits to authentication endpoints to prevent credential stuffing. "
    "Detection: login/auth endpoints missing throttling middleware.",
    "CWE-502 Deserialization of Untrusted Data: Never deserialize untrusted data with "
    "pickle. Prefer JSON with a strict schema or protobuf. "
    "Detection: pickle.loads / yaml.load called on external data.",
    "CWE-639 Insecure Direct Object Reference (IDOR): Always verify the requesting "
    "user owns or has access to the requested resource. "
    "Detection: resource IDs from URL/body used without ownership check.",
    "CWE-16 Security Misconfiguration (Headers): Set Strict-Transport-Security, "
    "X-Frame-Options, X-Content-Type-Options, and Referrer-Policy on all responses. "
    "Detection: missing security-header middleware in response pipeline.",
    "CWE-532 Insertion of Sensitive Information into Log File: Never log passwords, "
    "tokens, or PII. Use structured logging with field-level redaction. "
    "Detection: log statements containing auth headers, passwords, or user data.",
    "CWE-362 Race Condition: Protect shared state with locks or atomic operations. "
    "Use database transactions for multi-step read-modify-write patterns. "
    "Detection: counter increments or token refresh logic without locking.",
    "CWE-434 Unrestricted Upload of File with Dangerous Type: Validate uploaded file "
    "type and extension. Reject executable types; store outside web root. "
    "Detection: file upload handlers with no MIME or extension validation.",
    "CWE-611 XML External Entity (XXE): Disable external entity processing in XML "
    "parsers. Use a hardened parser configuration. "
    "Detection: lxml / ElementTree with external entity resolution enabled.",
    "CWE-918 Server-Side Request Forgery (SSRF): Validate and allowlist outbound URLs. "
    "Do not forward user-supplied URLs to internal services. "
    "Detection: HTTP client calls using user-controlled URL parameters.",
    "CWE-284 Improper Access Control: Enforce role-based or attribute-based access "
    "control on every protected resource. "
    "Detection: admin-only endpoints lacking authorization middleware.",
    "CWE-476 NULL Pointer Dereference: Validate return values before dereferencing. "
    "Use Optional types and null checks where APIs may return None. "
    "Detection: chained attribute access on values from external or optional sources.",
    "CWE-190 Integer Overflow: Use safe arithmetic or checked-integer libraries for "
    "untrusted numeric inputs. "
    "Detection: arithmetic on user-supplied integers without bounds checks.",
    "CWE-416 Use After Free: Avoid manual memory management; prefer garbage-collected "
    "or ownership-typed languages. In C/C++, audit all free() call sites. "
    "Detection: pointer reuse after deallocation in low-level code.",
    "CWE-306 Missing Authentication for Critical Function: Require authentication on "
    "all state-modifying endpoints. Unauthenticated access must be an explicit design "
    "decision. Detection: routes without auth middleware or session check.",
    "CWE-119 Buffer Overflow: Validate all buffer lengths before copy operations. "
    "Use safe string functions (strlcpy, snprintf). "
    "Detection: memcpy / strcpy calls with user-controlled length arguments.",
    "CWE-601 Open Redirect: Validate redirect targets against an allowlist of trusted "
    "domains. Reject or encode user-supplied redirect URLs. "
    "Detection: redirect() calls using unvalidated user input as the destination.",
    # --- Extended high-impact CWEs -------------------------------------------
    "CWE-787 Out-of-bounds Write: Validate all index and offset arithmetic before "
    "writing to buffers. Use bounded container APIs (e.g., std::vector::at). "
    "Detection: pointer arithmetic or array indexing without range checks.",
    "CWE-125 Out-of-bounds Read: Validate array indices and buffer positions before "
    "reading. Fuzz parsers that consume externally supplied data. "
    "Detection: slice/index operations on data whose length is user-controlled.",
    "CWE-78 OS Command Injection: Never construct shell commands from user input. "
    "Use subprocess with a list argument (never shell=True with interpolation). "
    "Detection: os.system / subprocess.run calls with string interpolation.",
    "CWE-77 Command Injection: Use parameterized commands or safe exec APIs. "
    "Allowlist permitted command arguments; reject anything outside the set. "
    "Detection: shell=True with user-derived command fragments.",
    "CWE-269 Improper Privilege Management: Drop privileges to the minimum required. "
    "Do not run services as root. Audit setuid/setgid usage. "
    "Detection: effective UID 0 or SID with SYSTEM during normal operations.",
    "CWE-400 Uncontrolled Resource Consumption (DoS): Apply limits on memory, CPU, "
    "and open connections per client. Use circuit breakers for downstream calls. "
    "Detection: unbounded loops, regex with catastrophic backtracking, no timeouts.",
    "CWE-295 Improper Certificate Validation: Never disable TLS certificate checks in "
    "production. Validate hostname and certificate chain. "
    "Detection: verify=False, ssl_verify=False, InsecureRequestWarning suppression.",
    "CWE-326 Inadequate Encryption Strength: Use AES-256, RSA-2048+, or ECC-256+ "
    "for all encryption. Retire DES, RC4, and RSA-1024. "
    "Detection: references to DES, RC2, RC4, Blowfish, or RSA key sizes <2048.",
    "CWE-327 Use of Broken or Risky Cryptographic Algorithm: Replace MD5 and SHA-1 "
    "with SHA-256+ for integrity. Never use ECB mode for block ciphers. "
    "Detection: hashlib.md5, hashlib.sha1, AES_ECB, DES usage.",
    "CWE-330 Use of Insufficiently Random Values: Use cryptographically secure "
    "random number generators (os.urandom, secrets module) for security-sensitive values. "
    "Detection: random.random() or time-seeded PRNG for token generation.",
    "CWE-338 Cryptographically Weak PRNG: Do not use math.random, java.util.Random, "
    "or Python's random module for security tokens. Use secrets.token_bytes(). "
    "Detection: non-cryptographic RNG seeded from predictable sources.",
    "CWE-613 Insufficient Session Expiration: Invalidate tokens and sessions after logout "
    "and on password change. Enforce absolute and idle session timeouts. "
    "Detection: JWT with no exp claim; sessions persisting after logout.",
    "CWE-614 Sensitive Cookie Without Secure Flag: Set Secure and SameSite=Strict on "
    "all session cookies. Never transmit session IDs over HTTP. "
    "Detection: Set-Cookie headers missing Secure or SameSite attribute.",
    "CWE-732 Incorrect Permission Assignment for Critical Resource: Restrict file and "
    "directory permissions to least privilege (600 for secrets, 640 for configs). "
    "Detection: chmod 777 or world-writable paths for config/key files.",
    "CWE-759 One-Way Hash Without a Salt: Always salt password hashes with a random, "
    "per-user salt before hashing. Unsalted hashes enable rainbow-table attacks. "
    "Detection: hash(password) without salt argument; MD5 of plain password.",
    "CWE-863 Incorrect Authorization: Verify both authentication (who) and authorization "
    "(what) on every request. Ownership checks must use server-side session data. "
    "Detection: authorization decisions based on client-supplied role/permission fields.",
    "CWE-943 NoSQL Injection: Sanitize all inputs used in MongoDB/CouchDB queries. "
    "Use typed query builders; never pass raw user dicts as query operators. "
    "Detection: '$where' or operator injection via user-controlled JSON keys.",
    "CWE-1004 Sensitive Cookie Without HttpOnly Flag: Set HttpOnly on session and auth "
    "cookies to prevent JavaScript access. Mitigates XSS-based cookie theft. "
    "Detection: Set-Cookie without HttpOnly for session or auth tokens.",
    "CWE-915 Mass Assignment / Improperly Controlled Object Modification: Use allowlists "
    "of permitted fields in request deserializers. Prevent clients from setting "
    "privileged fields. Detection: model.update(request.body) without field filtering.",
    "CWE-200 Information Exposure: Remove stack traces, debug information, and internal "
    "paths from error responses in production. Use opaque error IDs. "
    "Detection: exception traceback or internal path exposed in HTTP response body.",
    "CWE-311 Missing Encryption of Sensitive Data: Encrypt PII and credentials at rest "
    "using AES-256-GCM. Use TLS 1.2+ for data in transit. "
    "Detection: plaintext PII stored in database columns or flat files.",
    "CWE-548 Information Exposure Through Directory Listing: Disable directory listing "
    "on all web servers. Serve static files from a content delivery layer. "
    "Detection: Apache/Nginx autoindex on; Flask send_from_directory with browseable path.",
    "CWE-522 Insufficiently Protected Credentials: Store credentials encrypted at rest. "
    "Rotate API keys and service tokens on a fixed schedule. "
    "Detection: credentials in plaintext env files, hardcoded in source, or in git history.",
    "CWE-384 Session Fixation: Regenerate session IDs after successful authentication. "
    "Never accept externally supplied session tokens as authoritative. "
    "Detection: session ID unchanged across unauthenticated → authenticated transition.",
    "CWE-829 Inclusion of Functionality from Untrusted Control Sphere: Pin all third-party "
    "scripts by SRI hash. Avoid loading external JS at runtime without integrity checks. "
    "Detection: <script src> without integrity attribute; dynamic require() with CDN URLs.",
    "CWE-1236 CSV Injection: Sanitize data exported to CSV to prevent formula injection. "
    "Prefix cell values that start with =, +, -, @ with a single-quote. "
    "Detection: user-supplied strings written directly to CSV without sanitization.",
]

# ---------------------------------------------------------------------------
# Style conventions (~30 entries)
# ---------------------------------------------------------------------------

STYLE_CONVENTIONS: Final[list[str]] = [
    "Function length: Keep functions under 50 lines. Longer functions obscure intent "
    "and are hard to test in isolation. "
    "Detection: count non-blank, non-comment lines per function; flag >50.",
    "Naming — variables: Use descriptive snake_case names. Single-letter names (`x`, `tmp`) "
    "are only acceptable as loop indices or mathematical variables. "
    "Detection: variable names shorter than 3 characters in non-loop context.",
    "Naming — functions: Verb-noun naming for functions (`calculate_total`, `validate_user`). "
    "Avoid vague names like `do_stuff`, `process`, `handle`. "
    "Detection: function names without a verb prefix.",
    "Naming — consistency: Use one convention per module. Do not mix camelCase and snake_case. "
    "Classes use PascalCase; everything else uses snake_case in Python. "
    "Detection: camelCase identifiers outside class definitions in Python files.",
    "Cyclomatic complexity: Keep cyclomatic complexity below 10 per function. "
    "High complexity correlates with defect density and reduces testability. "
    "Detection: count decision points (if/elif/for/while/except/and/or per function) + 1.",
    "Nesting depth: Limit nesting to 3 levels. Deep nesting signals missing abstractions. "
    "Use early-return (guard clauses) and helper functions to flatten control flow. "
    "Detection: count indent levels; flag blocks nested >3 deep.",
    "Magic numbers: Replace numeric literals with named constants or enums. "
    "Magic numbers make code opaque and fragile under change. "
    "Detection: numeric literals other than 0, 1, -1 outside test files.",
    "Docstrings: Every public function, class, and module must have a docstring. "
    "One-line summary + Args + Returns for functions with parameters. "
    "Detection: `def` or `class` statement not followed by a string literal.",
    "TODO comments: Do not merge code with TODO/FIXME/HACK comments unless tracked in an issue. "
    "Unresolved TODOs indicate incomplete work. "
    "Detection: grep for TODO, FIXME, HACK, XXX, NOCOMMIT in changed lines.",
    "Dead code: Remove commented-out code blocks before merging. "
    "Use version control history instead of comment tombstones. "
    "Detection: blocks of consecutive commented lines (>3) that look like code.",
    "Import order: Follow isort conventions — stdlib, third-party, local, each group separated "
    "by a blank line. Use `from __future__ import annotations` at the top of every file. "
    "Detection: import groups not separated by blank lines; local before stdlib.",
    "Line length: Lines must not exceed 100 characters. Long lines reduce readability "
    "in side-by-side diff views and on narrow terminals. "
    "Detection: any line >100 characters in non-auto-generated files.",
    "Trailing whitespace: Remove trailing spaces and tabs. They pollute diffs and confuse "
    "some editors. Detection: lines ending with `\\s+$`.",
    "Type annotations: All public functions must have full type annotations on parameters "
    "and return types. Use `Optional[X]` or `X | None` for nullable values. "
    "Detection: function signatures missing annotations; missing return type.",
    "Exception handling: Avoid bare `except:` clauses. Catch specific exceptions and "
    "log or re-raise. Swallowing exceptions silently hides bugs. "
    "Detection: `except:` or `except Exception:` without a logged message or re-raise.",
    "Test coverage: Every new function should have a corresponding unit test. "
    "Added logic paths without tests are a coverage regression. "
    "Detection: new `def` statements in non-test files without matching test file changes.",
    "Constants: Module-level constants should be ALL_CAPS. "
    "Lowercase module-level names that are never reassigned should be constants. "
    "Detection: module-level assignments to lowercase names that are never mutated.",
    "String formatting: Use f-strings (Python ≥3.6) rather than `%` or `.format()`. "
    "F-strings are more readable and slightly faster. "
    "Detection: `%s` formatting or `.format(` calls in new code.",
    "Return types: Functions must not have multiple implicit `None` returns mixed with "
    "value returns. Use explicit `return None` or restructure with a single exit point. "
    "Detection: function body with both `return <value>` and bare `return` or fall-through.",
    "Dataclass vs dict: Use dataclasses or Pydantic models instead of plain dicts for "
    "structured data passed between functions. Typed structures are self-documenting. "
    "Detection: functions that return `dict[str, Any]` where a named type would fit.",
    "Global state: Avoid mutable module-level globals. Use dependency injection or "
    "configuration objects instead. "
    "Detection: module-level `list`, `dict`, or custom objects mutated in function bodies.",
    "Logging: Use structured logging (`structlog` or `logging`) instead of `print`. "
    "Print statements do not respect log levels or output routing. "
    "Detection: bare `print(` calls in non-script, non-test code.",
    "Assertions: Do not use `assert` for runtime validation in production code. "
    "Assertions are removed with `-O` and should be reserved for invariants in tests. "
    "Detection: `assert` statements outside test files.",
    "Comprehension clarity: List/dict/set comprehensions should fit on one line (≤80 chars). "
    "Nested comprehensions are almost always clearer as explicit loops. "
    "Detection: comprehensions with more than one `for` clause.",
    "Context managers: Use `with` statements for all resource acquisition (files, locks, "
    "DB connections). Manual `try/finally` for cleanup is error-prone. "
    "Detection: explicit `f.close()` or `lock.release()` outside a `finally` block.",
    "DRY principle: Avoid duplicated code blocks (>5 lines appearing more than once). "
    "Extract into a shared helper or utility function. "
    "Detection: identical or near-identical code segments repeated in the same diff.",
    "Class cohesion: Classes should have a single responsibility. God classes with >10 "
    "public methods often signal the need for decomposition. "
    "Detection: class with more than 10 public methods (not prefixed with `_`).",
    "Mutable default arguments: Never use mutable objects (list, dict, set) as default "
    "argument values. They are shared across all calls and cause subtle bugs. "
    "Detection: `def f(x=[])` or `def f(x={})` patterns.",
    "Shadowing builtins: Do not use builtin names (`list`, `dict`, `id`, `type`, `input`) "
    "as variable names. Shadowing them makes the builtins inaccessible in that scope. "
    "Detection: assignment to a name that matches a Python builtin.",
    "Unused imports: Remove all unused imports. They add noise and slow import time. "
    "Detection: imported names never referenced in the module body.",
]

# ---------------------------------------------------------------------------
# Report templates (~10 entries)
# ---------------------------------------------------------------------------

REPORT_TEMPLATES: Final[list[str]] = [
    "Template CRITICAL_REPORT: Open with 'MERGE BLOCKED — CRITICAL FINDINGS DETECTED'. "
    "List CRITICAL findings first, then HIGH. State that all CRITICAL issues require "
    "same-day remediation and must be escalated to the security team lead.",
    "Template HIGH_RISK_REPORT: Open with 'MERGE BLOCKED — HIGH SEVERITY FINDINGS'. "
    "Enumerate each HIGH finding with its CWE ID and line reference. "
    "State 7-day SLA for remediation before re-review.",
    "Template MEDIUM_RISK_REPORT: Open with 'CHANGES REQUESTED'. "
    "List MEDIUM findings with suggested fixes. "
    "Approve conditionally upon resolution within 30 days.",
    "Template LOW_RISK_REPORT: Open with 'APPROVED WITH NITS'. "
    "List LOW and informational findings as optional cleanup items. No merge block.",
    "Template CLEAN_REPORT: Open with 'APPROVED'. "
    "State no significant security or style issues found. "
    "List any trivial nits inline; encourage follow-up on deferred items.",
    "Template MIXED_SECURITY_STYLE: Lead with security section (risk-ordered findings), "
    "then style section (category-grouped findings). Provide separate recommendations "
    "for security and style tracks.",
    "Template DEPENDENCY_REPORT: List each CVE with its NVD score and affected version range. "
    "Recommend pinning to a patched version. Cross-reference SBOM and dependency audit logs.",
    "Template AUTH_CHANGE_REPORT: Apply elevated scrutiny to JWT, session, and credential "
    "findings first regardless of severity. Include references to OWASP ASVS Level 2.",
    "Template REFACTOR_REPORT: Security risk is typically low for refactoring PRs. "
    "Focus on complexity metrics, naming quality, and documentation completeness.",
    "Template STYLE_HEAVY_REPORT: No security findings. Detail each style category: "
    "complexity, naming, documentation, formatting, readability. "
    "Provide a code quality score summary.",
]

# ---------------------------------------------------------------------------
# Sample reviews (~20 synthetic PR review summaries for RAG context)
# ---------------------------------------------------------------------------

SAMPLE_REVIEWS: Final[list[str]] = [
    "Sample review PR-001 (SQL injection fix): PR adds parameterized queries to the user "
    "search endpoint. Security review found CWE-89 SQL injection in the original code at "
    "line 42. Fix uses SQLAlchemy text() with bound parameters. Risk: HIGH → resolved. "
    "Style: clean, no issues. Outcome: APPROVED after fix.",
    "Sample review PR-002 (auth middleware): Adds JWT validation middleware. "
    "Security: CWE-347 algorithm confusion risk — 'none' algorithm not rejected at line 18. "
    "CWE-613 no exp claim enforced. Style: missing docstrings on two public methods. "
    "Outcome: MERGE BLOCKED — critical JWT hardening required.",
    "Sample review PR-003 (file upload handler): Introduces S3 upload endpoint. "
    "Security: CWE-434 missing MIME type validation, CWE-22 path traversal via filename "
    "at line 67. Style: function too long (85 lines), magic number 5242880 (5MB). "
    "Outcome: MERGE BLOCKED — upload handler has exploitable path traversal.",
    "Sample review PR-004 (dependency update): Bumps requests from 2.27.1 to 2.31.0. "
    "Security: CVE-2023-32681 (CWE-937) in old version — patched in new version. "
    "No code changes. Style: N/A. Outcome: APPROVED — dependency fix.",
    "Sample review PR-005 (password reset flow): Refactors password reset token generation. "
    "Security: CWE-330 — uses random.randint() instead of secrets.token_urlsafe(). "
    "CWE-759 password hash has no per-user salt. Style: bare except clause at line 33. "
    "Outcome: MERGE BLOCKED — cryptographic weaknesses in token and hash generation.",
    "Sample review PR-006 (logging refactor): Replaces print statements with structlog. "
    "Security: CWE-532 — old code logged request.headers (contains Authorization) at "
    "line 91; new code redacts auth fields. Style: 4 print() calls removed, good. "
    "Outcome: APPROVED — security improvement and style cleanup.",
    "Sample review PR-007 (rate limiting): Adds token-bucket rate limiter to /api/login. "
    "Security: CWE-307 previously unmitigated. New middleware applies 5 req/min per IP. "
    "Style: constants (5, 60) should be named MAX_ATTEMPTS and WINDOW_SECONDS. "
    "Outcome: APPROVED WITH NITS — rate limiting is correct; name the constants.",
    "Sample review PR-008 (CSRF protection): Adds CSRF token to all POST forms. "
    "Security: fixes CWE-352 on /account/settings and /payment/checkout. Token is "
    "SameSite=Strict HttpOnly cookie. Style: no issues. "
    "Outcome: APPROVED — CSRF coverage is complete.",
    "Sample review PR-009 (XML parser): Integrates lxml to parse user-uploaded XML configs. "
    "Security: CWE-611 XXE — lxml defaults allow external entities. Missing "
    "resolve_entities=False and no_network=True flags at line 24. "
    "Style: missing type annotation on parse_config return type. "
    "Outcome: MERGE BLOCKED — XXE is trivially exploitable.",
    "Sample review PR-010 (admin dashboard): Adds /admin/users endpoint. "
    "Security: CWE-306 — endpoint has no authentication check. CWE-284 any logged-in "
    "user can access admin data. Style: God class AdminView has 14 public methods. "
    "Outcome: MERGE BLOCKED — unauthenticated admin access is a critical flaw.",
    "Sample review PR-011 (SSRF in webhook): Implements outbound webhook delivery. "
    "Security: CWE-918 — webhook URL is user-supplied with no allowlist; "
    "internal metadata service (169.254.169.254) reachable. Style: function length "
    "68 lines, nested 4 levels. Outcome: MERGE BLOCKED — SSRF to cloud metadata service.",
    "Sample review PR-012 (cookie attributes): Sets session cookie in new auth module. "
    "Security: CWE-614 missing Secure flag; CWE-1004 missing HttpOnly flag. "
    "Style: no issues. Outcome: CHANGES REQUESTED — add Secure and HttpOnly to cookie.",
    "Sample review PR-013 (refactor: extract service layer): Pure refactor PR. "
    "Security: no new attack surface. Style: cyclomatic complexity reduced from 18 to 6. "
    "Three functions still exceed 50 lines. Module-level mutable list 'cache'. "
    "Outcome: APPROVED WITH NITS — extract cache to injected dependency.",
    "Sample review PR-014 (pickle cache): Adds Redis-backed object cache using pickle. "
    "Security: CWE-502 — pickle.loads() on Redis-retrieved bytes is deserialization "
    "of potentially attacker-controlled data if Redis is accessible. "
    "Style: no docstring on CacheManager.__init__. "
    "Outcome: MERGE BLOCKED — replace pickle with JSON or msgpack.",
    "Sample review PR-015 (open redirect): Updates post-login redirect logic. "
    "Security: CWE-601 — 'next' query parameter used as redirect target without "
    "allowlist at line 55. Style: mixed f-string and .format() usage in same file. "
    "Outcome: MERGE BLOCKED — open redirect to arbitrary external URLs.",
    "Sample review PR-016 (CSV export): Adds user-data CSV export endpoint. "
    "Security: CWE-1236 CSV injection — user display names not sanitized before "
    "writing to CSV; formula injection via '=cmd|...' prefix possible. "
    "Style: line length 112 chars at line 77. Outcome: CHANGES REQUESTED — sanitize CSV cells.",
    "Sample review PR-017 (TLS config): Updates HTTPS server configuration. "
    "Security: CWE-326 — TLS 1.0 and TLS 1.1 still enabled in ssl_context. "
    "CWE-295 — verify=False in internal service client at line 14. "
    "Style: magic string 'TLSv1' should be constant. Outcome: MERGE BLOCKED — insecure TLS.",
    "Sample review PR-018 (role-based access control): Adds RBAC to /api/v2 endpoints. "
    "Security: CWE-863 — role sourced from JWT payload without server-side validation; "
    "client can self-elevate. Style: 6 TODO comments, no issue references. "
    "Outcome: MERGE BLOCKED — role must be fetched from server-side DB, not token.",
    "Sample review PR-019 (clean feature PR): Adds dark-mode toggle to user settings. "
    "Security: no security surface (UI preference only). Style: all functions documented, "
    "snake_case names, no magic numbers. Test coverage added. "
    "Outcome: APPROVED — clean, well-tested feature.",
    "Sample review PR-020 (NoSQL query): Migrates user search to MongoDB. "
    "Security: CWE-943 — search filter built from raw request JSON; '$where' operator "
    "injection possible. CWE-200 — error response leaks collection name and field list. "
    "Style: bare except catches all exceptions at line 88. "
    "Outcome: MERGE BLOCKED — NoSQL injection and information disclosure.",
]

# ---------------------------------------------------------------------------
# Collection names
# ---------------------------------------------------------------------------

COLLECTION_SECURITY = "security_patterns"
COLLECTION_STYLE = "style_conventions"
COLLECTION_TEMPLATES = "report_templates"
COLLECTION_REVIEWS = "sample_reviews"


# ---------------------------------------------------------------------------
# Core seeding logic
# ---------------------------------------------------------------------------


def _upsert_collection(
    client: ClientAPI,
    name: str,
    documents: list[str],
    backend: MockBackend,
    reset: bool = False,
) -> int:
    """Create or update a ChromaDB collection with the given documents.

    Args:
        client: ChromaDB client instance.
        name: Collection name.
        documents: Documents to upsert.
        backend: Backend used to compute embeddings.
        reset: If True, delete the collection before recreating it.

    Returns:
        Number of documents in the collection after seeding.
    """
    if reset:
        try:
            client.delete_collection(name)
            log.info("Deleted existing collection '%s'.", name)
        except Exception:  # noqa: BLE001 — collection may not exist
            pass

    collection = client.get_or_create_collection(name)

    ids = [f"{name}_{i:04d}" for i in range(len(documents))]
    embeddings = backend.embed(documents)

    collection.upsert(documents=documents, ids=ids, embeddings=embeddings)
    count = collection.count()
    log.info("Collection '%s': %d documents.", name, count)
    return count


def seed(chroma_dir: str | Path, reset: bool = False) -> dict[str, int]:
    """Populate all four ChronoAgent memory collections.

    Args:
        chroma_dir: Path to the ChromaDB persistence directory.
        reset: If True, wipe existing collections before seeding.

    Returns:
        Dict mapping collection name to document count after seeding.
    """
    path = Path(chroma_dir)
    path.mkdir(parents=True, exist_ok=True)

    client: ClientAPI = chromadb.PersistentClient(path=str(path))
    backend = MockBackend(seed=0)

    log.info("Seeding ChromaDB at %s …", path.resolve())

    counts: dict[str, int] = {}
    for name, docs in [
        (COLLECTION_SECURITY, SECURITY_PATTERNS),
        (COLLECTION_STYLE, STYLE_CONVENTIONS),
        (COLLECTION_TEMPLATES, REPORT_TEMPLATES),
        (COLLECTION_REVIEWS, SAMPLE_REVIEWS),
    ]:
        counts[name] = _upsert_collection(client, name, docs, backend, reset=reset)

    total = sum(counts.values())
    log.info("Seed complete. Total documents: %d.", total)
    return counts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (uses sys.argv if None).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="seed_memory",
        description="Seed ChronoAgent ChromaDB memory collections.",
    )
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_data",
        help="Path to ChromaDB persistence directory (default: ./chroma_data).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collections before seeding.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the seed script.

    Args:
        argv: Argument list (uses sys.argv if None).
    """
    args = _parse_args(argv)
    counts = seed(chroma_dir=args.chroma_dir, reset=args.reset)
    for name, count in counts.items():
        print(f"  {name}: {count} documents")
    sys.exit(0)


if __name__ == "__main__":
    main()
