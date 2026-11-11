# Security Policy

## Supported Versions

Only the latest release of hfgo is supported with security updates. Due to the
fast-moving nature of the upstream Hugging Face Inference API, older versions
may be incompatible with current API behavior and are not maintained.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use
[GitHub Private Vulnerability Reporting](https://github.com/Kardbord/hfgo/security/advisories/new)
to report a vulnerability. This ensures the report is only visible to
maintainers until a fix is available.

### What to Report

Examples of reportable vulnerabilities include:

- Exposure or leakage of API tokens or credentials
- Injection or manipulation of user-controlled inputs
- Dependency vulnerabilities that affect the SDK at runtime
- Unsafe deserialization or data handling

## Disclosure Policy

- **Acknowledgement**: We aim to acknowledge receipt of your report within
  **48 hours**.
- **Resolution**: We aim to provide a fix within **90 days** of the initial
  report. If a fix requires more time, we will communicate an updated timeline.
- **Disclosure**: We request that reporters allow coordinated disclosure so
  that a fix can be released before details are made public.

## Scope

This policy covers the hfgo Go module and its direct dependencies. Issues
upstream in the Hugging Face Inference API itself should be reported to
[Hugging Face](https://huggingface.co/).
