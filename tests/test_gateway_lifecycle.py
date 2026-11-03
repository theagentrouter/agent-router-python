"""Starting and stopping the gateway, using real child processes, locks, files and HTTP requests."""

import json
import os
import subprocess
import sys
import time

import httpx
import pytest

import theagentrouter as ar
from theagentrouter import anthropic_format, exceptions, local_gateway, routing, transport
from theagentrouter.compat import litellm

from .test_local_gateway import env as gateway_env
from .test_local_gateway import fake as gateway_fake

env = gateway_env
fake = gateway_fake


def test_interrupted_start_kills_and_reaps_the_child(env, tmp_path, monkeypatch):
    program = tmp_path / "aigw"
    program.write_text(f"#!{sys.executable}\nimport time\nprint('listening on', flush=True)\ntime.sleep(60)\n")
    program.chmod(0o700)
    local_gateway._private_dir(local_gateway.folder())
    monkeypatch.setattr(local_gateway, "_binary", lambda: str(program))
    started = []
    popen = subprocess.Popen

    def record(*args, **kwargs):
        proc = popen(*args, **kwargs)
        started.append(proc)
        return proc

    def interrupt(port):
        state = json.loads((local_gateway.folder() / "state.json").read_text())
        assert state["starting"] is True and state["pid"] == started[0].pid
        raise KeyboardInterrupt

    monkeypatch.setattr(local_gateway.subprocess, "Popen", record)
    monkeypatch.setattr(local_gateway, "_listening", interrupt)
    with pytest.raises(KeyboardInterrupt):
        local_gateway._launch(
            local_gateway.folder(),
            dict(os.environ),
            local_gateway.folder() / "aigw.log",
            ["openai"],
            ["openai/gpt-4o-mini"],
            {},
            12345,
            12346,
        )
    assert started[0].poll() is not None
    assert not (local_gateway.folder() / "state.json").exists()


def test_stop_reaps_child_and_removes_key_files(env, monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(local_gateway, "_runtime_dir", lambda: runtime)
    local_gateway._private_dir(runtime)
    keys = local_gateway.folder() / "aigw-state" / "old-run"
    keys.mkdir(parents=True)
    (keys / "secret.yaml").write_text("acme-old-key")
    program = tmp_path / "aigw"
    program.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    program.chmod(0o700)
    proc = subprocess.Popen([str(program)], start_new_session=True)
    local_gateway._write_state({"pid": proc.pid, "port": 9, "starting": True})
    started = time.monotonic()
    try:
        local_gateway.stop()
        assert time.monotonic() - started < 3
        assert not local_gateway._alive(proc.pid)
        assert not keys.exists() and not runtime.exists()
        assert not (local_gateway.folder() / "state.json").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_start_lock_times_out_between_processes(env, monkeypatch):
    local_gateway._private_dir(local_gateway.folder())
    lock = local_gateway.folder() / "lock"
    code = "import fcntl,time,sys; f=open(sys.argv[1],'a+'); fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); time.sleep(60)"
    proc = subprocess.Popen([sys.executable, "-c", code, str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "locked"
        monkeypatch.setattr(local_gateway, "LOCK_TIMEOUT", 0.1)
        with pytest.raises(exceptions.Timeout, match="startup lock"), local_gateway._lock():
            pytest.fail("acquired a lock held by another process")
    finally:
        proc.terminate()
        proc.wait()
        proc.stdout.close()


def test_runtime_symlink_is_rejected_before_start(env, tmp_path, monkeypatch):
    target = tmp_path / "other"
    target.mkdir()
    runtime = tmp_path / "runtime"
    runtime.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(local_gateway, "_runtime_dir", lambda: runtime)
    monkeypatch.setattr(local_gateway, "_busy_internal", lambda: None)
    monkeypatch.setattr(local_gateway, "_launch", lambda *args: pytest.fail("launched with an unsafe runtime"))
    with pytest.raises(exceptions.APIConnectionError, match="unsafe local gateway directory"):
        local_gateway._start(["openai"], ["openai/gpt-4o-mini"])


def test_runtime_and_state_are_private(env, tmp_path):
    directory = tmp_path / "runtime"
    directory.mkdir(mode=0o755)
    local_gateway._private_dir(directory)
    assert directory.stat().st_mode & 0o777 == 0o700
    local_gateway._write_state({"pid": 42, "port": 9})
    state = local_gateway.folder() / "state.json"
    assert state.stat().st_mode & 0o777 == 0o600


def test_rotated_provider_key_is_reported_before_a_request(env, monkeypatch):
    state = {
        "pid": 42,
        "port": 9,
        "providers": ["openai"],
        "models": ["openai/gpt-4o-mini"],
        "credential_fingerprints": local_gateway._credential_fingerprints(["openai"], dict(os.environ)),
    }
    assert "sk-test" not in json.dumps(state)
    monkeypatch.setenv("OPENAI_API_KEY", "acme-rotated-key")
    monkeypatch.setattr(local_gateway, "running", lambda: state)
    with pytest.raises(exceptions.AuthenticationError, match="credentials changed"):
        routing.resolve("openai/gpt-4o-mini")


def test_missing_previous_provider_key_does_not_stop_shared_gateway(env, monkeypatch):
    state = {
        "providers": ["anthropic", "openai"],
        "models": ["openai/gpt-4o-mini"],
        "versions": local_gateway.versions(),
    }
    monkeypatch.setattr(local_gateway, "running", lambda: state)
    monkeypatch.setattr(local_gateway, "stop", lambda *args: pytest.fail("stopped the shared gateway"))
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(exceptions.AuthenticationError, match="every provider already running"):
        local_gateway.ensure("openai/gpt-4o")


@pytest.mark.parametrize("region", ["us-east-1\nextra: injected", 'us-east-1"', "{region}", "../us-east-1"])
def test_aws_region_cannot_inject_yaml(region):
    with pytest.raises(exceptions.BadRequestError, match="AWS region name"):
        local_gateway.config_yaml(["bedrock"], {"AWS_REGION": region}, port=1975)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_provider_base_url_overrides_are_in_the_generated_config(provider):
    config = local_gateway.config_yaml(
        [provider], {f"{provider.upper()}_BASE_URL": "http://acme.example:8123/v1"}, port=1975
    )
    assert "hostname: acme.example" in config and "port: 8123" in config
    assert "api.openai.com" not in config and "api.anthropic.com" not in config


def test_already_invalid_retry_does_not_disable_other_routes(env, monkeypatch):
    ar.route("working", primary="openai/gpt-4o-mini")
    path = env / "routes.json"
    raw = json.loads(path.read_text())
    raw["broken"] = {**raw["working"], "retry": {"on": [42]}}
    path.write_text(json.dumps(raw))
    before = path.read_bytes()
    with pytest.warns(UserWarning, match="broken.*invalid and disabled"):
        assert set(ar.list_routes()) == {"working"}
    assert path.read_bytes() == before


def test_empty_streamed_tool_arguments_form_a_json_object():
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "acme-call", "name": "lookup", "input": {}},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    parser = anthropic_format.StreamParser()
    chunks = [chunk for event in events for chunk in parser.feed(json.dumps(event))]
    arguments = "".join(
        call.function.arguments
        for chunk in chunks
        for choice in chunk.choices
        for call in choice.delta.tool_calls or []
    )
    assert json.loads(arguments) == {}


def test_module_defaults_apply_to_embedding_and_responses(env, fake, monkeypatch):
    monkeypatch.setattr(litellm, "api_base", fake.url)
    monkeypatch.setattr(litellm, "api_key", "acme-module-key")
    assert litellm.embedding(model="openai/text-embedding-3-small", input="acme").data
    assert litellm.responses(model="openai/gpt-4o-mini", input="acme").output_text
    assert all(headers["authorization"] == "Bearer acme-module-key" for _, headers, _ in fake.requests)


async def test_async_module_defaults_apply_to_embedding_and_responses(env, fake, monkeypatch):
    monkeypatch.setattr(litellm, "api_base", fake.url)
    monkeypatch.setattr(litellm, "api_key", "acme-module-key")
    assert (await litellm.aembedding(model="openai/text-embedding-3-small", input="acme")).data
    assert (await litellm.aresponses(model="openai/gpt-4o-mini", input="acme")).output_text
    assert all(headers["authorization"] == "Bearer acme-module-key" for _, headers, _ in fake.requests)
    await ar.aclose()


def test_target_repr_hides_explicit_authentication_headers():
    target = routing.Target(
        "gateway", "openai", "http://acme.example/v1", None, "acme", "openai", {"Authorization": "Bearer acme-secret"}
    )
    assert "acme-secret" not in repr(target)


def test_exception_constructor_keeps_litellm_order_and_response_headers():
    response = httpx.Response(429, headers={"retry-after": "3", "x-request-id": "acme-request"})
    error = exceptions.RateLimitError("limited", "openai", "gpt-4o-mini", response)
    assert (error.model, error.llm_provider, error.status_code) == ("gpt-4o-mini", "openai", 429)
    assert error.headers["x-request-id"] == "acme-request"
    generic = exceptions.APIError(503, "unavailable", "openai", "gpt-4o-mini")
    assert generic.status_code == 503 and generic.model == "gpt-4o-mini"


def test_added_router_methods_work(env, fake, monkeypatch):
    monkeypatch.setattr(litellm, "api_base", fake.url)
    model_list = [{"model_name": "embedding", "litellm_params": {"model": "openai/text-embedding-3-small"}}]
    # The methods accept a provider/model directly, just as completion does.
    router = litellm.Router(model_list=[])
    router.model_list = model_list
    assert router.embedding(model="openai/text-embedding-3-small", input="acme").data
    returned = router.get_model_list()
    returned[0]["model_name"] = "changed"
    assert router.model_list[0]["model_name"] == "embedding"


def test_unknown_model_cost_is_explicitly_unpriced():
    with pytest.warns(UserWarning, match="No price is known"):
        assert ar.completion_cost(model="acme/unpriced-lifecycle-test", prompt_tokens=100) == 0


@pytest.mark.parametrize("model,count", [("err-500", 3), ("err-401", 1), ("err-budget", 1)])
def test_retries_are_bounded_and_authentication_and_budget_errors_are_not_retried(env, fake, monkeypatch, model, count):
    monkeypatch.setattr(transport, "_retry_delay", lambda *args: 0)
    with pytest.raises(exceptions.APIError):
        ar.completion(
            model=f"openai/{model}", messages=[{"role": "user", "content": "acme"}], api_base=fake.url, num_retries=2
        )
    assert len(fake.requests) == count


async def test_async_retries_use_the_same_limit(env, fake, monkeypatch):
    monkeypatch.setattr(transport, "_retry_delay", lambda *args: 0)
    with pytest.raises(exceptions.InternalServerError):
        await ar.acompletion(
            model="openai/err-500", messages=[{"role": "user", "content": "acme"}], api_base=fake.url, num_retries=2
        )
    assert len(fake.requests) == 3
    await ar.aclose()


def test_partial_stream_is_not_retried(env, fake, monkeypatch):
    monkeypatch.setattr(transport, "_retry_delay", lambda *args: 0)
    stream = ar.completion(
        model="openai/stream-error",
        messages=[{"role": "user", "content": "acme"}],
        api_base=fake.url,
        stream=True,
        num_retries=2,
    )
    assert next(stream).choices[0].delta.content == "partial"
    with pytest.raises(exceptions.RateLimitError) as caught:
        list(stream)
    assert caught.value.response is not None and len(fake.requests) == 1


@pytest.mark.parametrize("name", ["malformed-json", "malformed-shape"])
def test_malformed_native_messages_raise_sdk_errors(env, fake, name):
    with pytest.raises(exceptions.APIResponseValidationError):
        ar.anthropic.messages.create(
            model=f"anthropic/{name}", messages=[{"role": "user", "content": "acme"}], max_tokens=10, api_base=fake.url
        )


def test_stop_leaves_another_program_with_the_gateways_old_number(env):
    # The gateway exited and its process number went to another program before stop() ran.
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    local_gateway._write_state({"pid": other.pid, "port": 9})
    try:
        local_gateway.stop()
        assert other.poll() is None
        assert not (local_gateway.folder() / "state.json").exists()
    finally:
        other.kill()
        other.wait()


def test_stop_does_not_wait_for_another_parents_zombie(env):
    # A live parent deliberately holds an exited child without calling waitpid.
    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True); print(p.pid,flush=True); sys.stdin.readline(); p.wait()"
    parent = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        child = int(parent.stdout.readline())
        time.sleep(0.2)
        start = time.monotonic()
        local_gateway.stop({"pid": child})
        assert time.monotonic() - start < 3
    finally:
        parent.communicate("reap\n", timeout=5)
