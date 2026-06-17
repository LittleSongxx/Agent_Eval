# Security Policy

## Supported Versions

This project is currently pre-1.0. Security fixes are handled on the main development line unless release branches are introduced later.

## Reporting a Vulnerability

Please report security issues privately to the maintainers instead of opening a public issue.

Include:

- Affected version or commit.
- Reproduction steps.
- Impact assessment.
- Whether secrets, uploaded files, evaluation reports, or customer data may be exposed.

Do not include real API keys, bearer tokens, private documents, or customer data in the report body. Use redacted examples where possible.

## Sensitive Data

The platform can store sensitive information:

- LLM API keys.
- Endpoint authorization headers.
- Uploaded RAG documents.
- Evaluation datasets.
- Model responses, retrieved contexts, tool traces, and judge reasons.

Current protections include masked API responses for LLM keys and endpoint authorization fields. For production deployment, add:

- Authentication and authorization.
- TLS.
- Database field encryption or KMS-backed secret storage.
- Audit logs.
- Backup encryption.
- Upload file access control.
- Secret rotation.

## Deployment Warning

The default local setup is intended for trusted development environments. Do not expose the backend directly to the public Internet without authentication, authorization, TLS, rate limiting, and secret management.

## English Summary

Report vulnerabilities privately. Treat API keys, authorization headers, datasets, uploaded files, and evaluation reports as sensitive data.
