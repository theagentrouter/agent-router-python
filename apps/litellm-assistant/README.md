# LiteLLM assistant

Plain LiteLLM code, with no agent framework: [`src/assistant/app.py`](src/assistant/app.py) asks a question, streams an answer with the loop from LiteLLM's README, streams asynchronously, runs one tool round and makes embeddings. It runs on LiteLLM or on Agent Router's SDK; [`src/assistant/__init__.py`](src/assistant/__init__.py) holds the whole migration, one `install_as_litellm()` call when the SDK is installed.

```sh
pip install -e ".[litellm,test]"            # on LiteLLM
pip install -e ../.. -e ".[test]"            # on Agent Router's SDK, from this repository
APP_SDK=litellm pytest                       # or APP_SDK=theagentrouter
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python examples/live.py
```

What the tests cover is in [`../README.md`](../README.md).
