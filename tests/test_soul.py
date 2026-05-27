from src.agents.soul import ensure_soul, load_soul


def test_soul_helpers_use_existing_agent_folder_without_overwriting(tmp_path):
    root = tmp_path / "agents"
    path = ensure_soul("Maria", {"personality": "Warm", "goals": ["Host well"]}, root)
    path.write_text("hand edited soul", encoding="utf-8")

    ensure_soul("Maria", {"personality": "Different"}, root)

    assert load_soul("Maria", root=root) == "hand edited soul"
