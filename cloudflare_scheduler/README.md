# NW FloodWatch Cloudflare Scheduler

This Worker is an independent scheduler for the GitHub Actions workflow:

- repository: `pgiordana/NW-HydroClimate-FloodWatch`
- workflow: `.github/workflows/nw-floodwatch-daily.yml`
- target branch: `main`
- intended local schedule: 10:17 and 11:00 Europe/Rome

## Why four UTC cron expressions?

Cloudflare Cron Triggers use UTC. Europe/Rome changes between CET and CEST, so `wrangler.toml` registers both UTC candidates for each desired local time. The Worker converts the trigger timestamp back to `Europe/Rome` and only dispatches when the local time is exactly `10:17` or `11:00`. The other seasonal candidate exits without doing anything.

## Duplicate protection

Before dispatching GitHub, the Worker:

1. reads `https://nw-floodwatch.pages.dev/data/latest.json` and exits if `issue_date` is already today in Europe/Rome;
2. checks recent runs of `nw-floodwatch-daily.yml` and exits if one created today is already `queued` or `in_progress`;
3. otherwise calls GitHub `workflow_dispatch` on `main`.

The public-site freshness check is fail-open: if the site cannot be read, production is not suppressed. The GitHub run-state check is authenticated and must succeed before a dispatch is sent.

## Required Cloudflare secret

Create a Worker secret named:

`GITHUB_ACTIONS_TOKEN`

Use a fine-grained GitHub personal access token scoped only to this repository and with the minimum permission required to run Actions workflows (`Actions: Read and write`; add repository metadata/read access if GitHub requires it). Never store the token in the repository or in `wrangler.toml`.

## Deployment

From the `cloudflare_scheduler` directory, the Worker can be deployed with Wrangler after authenticating to Cloudflare:

```bash
npx wrangler deploy
```

Then set the secret interactively:

```bash
npx wrangler secret put GITHUB_ACTIONS_TOKEN
```

The same files can also be imported into Cloudflare Workers Builds from the `cloudflare-scheduler-v1` branch, using `cloudflare_scheduler` as the project root.

## Manual HTTP endpoint

A normal HTTP GET to the Worker returns only a health/status JSON. It does not trigger GitHub. Production starts only from scheduled events, so exposing the Worker URL does not create a public manual-trigger endpoint.
