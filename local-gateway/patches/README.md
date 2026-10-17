# Local aigw patches

The builder applies the `aigw-` patches to upstream Agent Router v1.2.0, in this order, and the `envoy-gateway-` patch to the Envoy Gateway it builds against.

**`aigw-v1.2.0-client-secrets.patch`.** The server-authentication fix needs this patch in addition to the SDK configuration change. Upstream `aigw run` v1.2.0 passes original Secrets into the translator but emits only generated Secrets. Envoy Gateway therefore cannot find the client-key Secret referenced by the exported authentication policy. The patch retains original Secrets and adds an upstream regression test. It does not change internal listen addresses.

**`aigw-v1.2.0-mcp-session-seed.patch`.** `aigw run` always starts an MCP proxy, and upstream v1.2.0 gives it no seed, so its session keys come from a public default. The patch adds `--mcp-session-encryption-seed`, also read from `AIGW_MCP_SESSION_ENCRYPTION_SEED`, and passes it to the external processor; without it, behavior is unchanged. The local gateway sets a random seed at each start. The patch updates the `aigw run --help` test and adds one for the passed seed.

**`aigw-v1.2.0-reload-on-sighup.patch`.** Upstream `aigw run` reads its config once, so the local gateway had to restart for a new route or model, and other programs' requests were cut off. With the patch, SIGHUP makes `aigw run` translate the config file again: the extension server reads the new objects, the external processor gets its new config (it checks every five seconds, so a reload waits six), and Envoy Gateway, which watches its resources folder, updates Envoy without restarting it. Values `aigw` reads from its own environment, such as provider keys, cannot change this way. `aigw run --help` says that SIGHUP reloads, which is how the SDK knows a companion can. The patch adds an upstream test for the reload.

**`aigw-v1.2.0-loopback-listeners.patch`.** `aigw run` serves one computer, but upstream binds the MCP proxy and the external processor's admin server to every address and lets Envoy Gateway start a metrics server on 19001. The patch passes `127.0.0.1` addresses for both (the external processor gains `-adminAddr`, which overrides `-adminPort`; Kubernetes deployments are unchanged), turns Envoy Gateway's metrics off, and enables `EnvoyPatchPolicy`, through which the SDK's config moves Envoy's listener and readiness listener to `127.0.0.1`. It adds an upstream test for each address.

**`envoy-gateway-v1.9.2-loopback-listeners.patch`.** Envoy Gateway v1.9.2, which `aigw` v1.2.0 builds against, binds its xDS server (18000), wasm cache (18002) and file-provider health probe (8081) to every address. The patch binds them to `127.0.0.1`. The builder clones Envoy Gateway at the pinned `v1.9.2` commit, applies it, and builds `aigw` against the copy with a `replace` directive. Each changed file says it was modified. Upstream needs these addresses to stay configurable for Kubernetes, so an upstream change would be an option rather than this one-line move.

Build from the repository root with Git and Go 1.27.1 available:

```sh
python scripts/build_review_gateway.py
pip install -e ./local-gateway
```

The builder clones upstream v1.2.0 into a temporary directory, checks commit `daa9f891a8afcb18576d4593ad18870d1d18abac`, applies the patch, runs its regression test and builds the ignored companion binary. CI uses the same builder on each supported binary platform. No existing checkout is changed. If the pinned source is already available, --source /path/to/upstream reads its committed Git objects into a separate temporary clone; it does not copy uncommitted changes. macOS builds supply explicit deployment and C compiler flags for macOS 13. Linux builds require readelf and reject glibc symbol requirements newer than 2.36 before packaging. The manifest's binary checksums still identify the unmodified upstream release; the patched source build has a different checksum. The builder refuses a changed release pin until the patch is reviewed or removed.

Once an upstream release includes Secret retention, remove the patch and source builder, update the release pin/checksums, and restore `scripts/fetch_aigw.py` in the packaging workflow.
