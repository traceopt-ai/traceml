from traceml.transport import distributed


def test_ddp_info_does_not_require_torch(monkeypatch):
    monkeypatch.setattr(distributed, "_load_torch", lambda: None)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    assert distributed.get_ddp_info() == (False, -1, 1)


def test_ddp_info_uses_env_when_torch_is_unavailable(monkeypatch):
    monkeypatch.setattr(distributed, "_load_torch", lambda: None)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert distributed.get_ddp_info() == (True, 0, 4)
