# Moving an ADK project from LiteLLM to Agent Router's SDK

This app started on LiteLLM. In a project of your own, the move is two changes.

1. In the dependencies, replace `litellm` with `theagentrouter`:

   ```diff
    dependencies = [
        "google-adk>=1.16,<3",
   -    "litellm>=1.84,<2",
   +    "theagentrouter",
        "mcp>=1.26,<2",
    ]
   ```

2. In the package's `__init__.py`, before any ADK import, replace LiteLLM's imports:

   ```python
   from theagentrouter.compat import install_as_litellm

   install_as_litellm()
   ```

Then reinstall, so LiteLLM leaves the environment. The agent, the runner, ADK's `LiteLlm` adapter, the MCP toolset, the tools, the task and the model names stay as they are.

This app keeps both SDKs possible, so its `__init__.py` calls `install_as_litellm()` only when Agent Router's SDK is installed; a project that has moved needs only the two lines above.

Two LiteLLM arguments to leave out of ADK's model options: `num_retries` (define a route with a retry instead) and `fallbacks` (a route's fallbacks). [Coming from LiteLLM](../../docs/coming-from-litellm.md) has the full list.
