# Security

## Data and credential boundaries

This project analyzes public SEC filings. Do not place confidential company,
customer, or personal data in the filing corpus, prepared JSON, prompts, logs, or
evaluation sets.

Keep optional Hugging Face configuration in environment variables or an
untracked local `.env` / `.streamlit/secrets.toml`. The default TinyLlama model
does not require an API key. Never put real secrets in `.env.example`, source
code, prepared data, screenshots, issues, or commits. The presentation layer
does not display environment variables and sanitizes provider errors before
rendering them.

## Deployment checklist

1. Confirm `.env`, `.streamlit/secrets.toml`, local indexes, and downloaded
   filings are excluded from version control.
2. Use a least-privilege provider key with spending and rate limits.
3. Restrict outbound traffic to required providers and official SEC domains.
4. Review prepared JSON for accidental secrets before deployment.
5. Pin and scan dependencies in the deployment environment.
6. Do not expose Streamlit directly to the public internet without
   authentication, TLS, and normal platform protections.

## Reporting

Report a suspected vulnerability privately to the repository owner. Include
reproduction steps and impact, but do not include live credentials or sensitive
data. Rotate any credential immediately if it may have been exposed.
