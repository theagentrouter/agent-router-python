# Acme team onboarding

This is fictional example product documentation.

## First setup

Create an account, verify the account email, and create a project named Acme.

An administrator invites Users from Settings → Users. Invitations expire after seven days.

## API integration

Create an API key from Settings → API keys. Save it in the application's environment and never include it in a support ticket.

Start with the example request from the integration guide. HTTP 401 means the key is missing or invalid. HTTP 429 means the request limit was reached.

For HTTP 429, wait for the Retry-After value in seconds before retrying. Use exponential backoff if that header is absent.

## Getting help

Include the request ID, timestamp with timezone, HTTP status, and a redacted request example when contacting support.
