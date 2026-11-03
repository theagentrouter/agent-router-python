# SPDX-License-Identifier: Apache-2.0
"""LiteLLM's Router, unchanged apart from the import: model groups with fallbacks and retries.

The Router's groups become routes in the local gateway, which does the retries and the fallback.

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python examples/from_litellm/litellm_router.py
"""

from theagentrouter.compat import litellm  # was: import litellm

router = litellm.Router(
    model_list=[
        {"model_name": "router-chat", "litellm_params": {"model": "openai/gpt-does-not-exist"}},
        {"model_name": "router-backup", "litellm_params": {"model": "anthropic/claude-haiku-5-5"}},
    ],
    fallbacks=[{"router-chat": ["router-backup"]}],
    num_retries=1,
)
resp = router.completion(model="router-chat", messages=[{"role": "user", "content": "Say hello."}])
print(resp.choices[0].message.content)
print("answered by:", resp._hidden_params["served_by"])
