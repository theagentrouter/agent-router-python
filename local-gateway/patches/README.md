# Local aigw patches

The builder applies these to upstream v1.2.0, in this order.

**`aigw-v1.2.0-client-secrets.patch`.** The server-authentication fix needs this patch in addition to the SDK configuration change. Upstream `aigw run` v1.2.0 passes original Secrets into the translator but emits only generated Secrets. Envoy Gateway therefore cannot find the client-key Secret referenced by the exported authentication policy. The patch retains original Secrets and adds an upstream regression test. It does not change internal listen addresses.

**`aigw-v1.2.0-mcp-session-seed.patch`.** `aigw run` always starts an MCP proxy, and upstream v1.2.0 gives it no seed, so its session keys come from a public default. The patch adds `--mcp-session-encryption-seed`, also read from `AIGW_MCP_SESSION_ENCRYPTION_SEED`, and passes it to the external processor; without it, behavior is unchanged. The local gateway sets a random seed at each start. The patch updates the `aigw run --help` test and adds one for the passed seed.

Build from the repository root with Git and Go 1.27.1 available:

```sh
python scripts/build_review_gateway.py
pip install -e ./local-gateway
```

The builder clones upstream v1.2.0 into a temporary directory, checks commit `daa9f891a8afcb18576d4593ad18870d1d18abac`, applies the patch, runs its regression test and builds the ignored companion binary. CI uses the same builder on each supported binary platform. No existing checkout is changed. If the pinned source is already available, --source /path/to/upstream reads its committed Git objects into a separate temporary clone; it does not copy uncommitted changes. macOS builds supply explicit deployment and C compiler flags for macOS 13. Linux builds require readelf and reject glibc symbol requirements newer than 2.36 before packaging. The manifest's binary checksums still identify the unmodified upstream release; the patched source build has a different checksum. The builder refuses a changed release pin until the patch is reviewed or removed.

Once an upstream release includes Secret retention, remove the patch and source builder, update the release pin/checksums, and restore `scripts/fetch_aigw.py` in the packaging workflow.
