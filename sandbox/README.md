# Expert 802 isolated code runner

This is a separate self-hosted service. Do not run arbitrary user code inside the Vercel function.

## Requirements

- Linux host with Docker Engine and permission to run Docker.
- Python 3.10+ for the small HTTP service.
- Preloaded images:
  ```bash
  docker pull python:3.11-slim
  docker pull node:22-alpine
  ```

## Start

Generate a long random shared secret (at least 32 characters), then run:

```bash
export EXPERT802_SANDBOX_TOKEN='replace-with-a-random-secret-of-at-least-32-characters'
python sandbox/runner.py --host 127.0.0.1 --port 8790
```

Keep the runner bound to localhost. Put a trusted TLS reverse proxy in front of it if it must be reached by Vercel. Do not expose the raw HTTP service to the public internet.

Set these Vercel environment variables:

- `EXPERT802_SANDBOX_URL=https://your-private-runner.example/run`
- `EXPERT802_SANDBOX_TOKEN` — the same random secret configured on the runner.

The app probes `/health` and only enables the Code Lab test button when the runner responds.

## Allowed operations

- `python_unittest`: runs `python -B -m unittest discover -s tests -v` in `python:3.11-slim`.
- `node_check`: runs `node --check` on one supplied JavaScript file in `node:22-alpine`.

There is no shell command input, package installation, network access from containers, write access to the project snapshot, or unrestricted execution API.

## Isolation controls

Each request uses a fresh Docker container with network disabled, read-only root filesystem and source mount, unprivileged UID, dropped Linux capabilities, no-new-privileges, 768 MB memory, one CPU, 64 process limit, 64 MB temporary filesystem and 45-second timeout. The HTTP endpoint requires bearer authentication. Output is truncated.

This is defense-in-depth, not a guarantee against every container escape. Keep Docker patched, isolate the host, restrict who can reach the service, and never run the Docker daemon on a shared untrusted machine.
