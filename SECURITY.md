# Security policy

## Secrets

Never commit `.streamlit/secrets.toml`, `.env`, API responses containing
credentials, or copied terminal output that includes a key. Production secrets
must be configured in the hosting platform's secret manager.

If a key is exposed, revoke it at the provider before removing it from Git
history. Deleting the visible file is not sufficient.

## Data boundaries

- External news titles and summaries are untrusted input.
- Uploaded JSON and PDF files are parsed, not executed.
- Provider errors shown to users are sanitized.
- AI POST requests are not automatically retried to avoid duplicate cost.

Please report security issues privately to the repository owner rather than
opening a public issue containing credentials or private data.
