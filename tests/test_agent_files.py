from src.agents.files import AgentFiles


def test_agent_files_create_and_reload_live_context(tmp_path):
    files = AgentFiles(tmp_path / "agents")
    bio = {"name": "Maria", "personality": "Warm", "goals": ["Welcome people"]}

    files.ensure(bio)
    folder = tmp_path / "agents" / "maria"
    assert (folder / "SOUL.md").exists()
    assert (folder / "KNOWLEDGE.md").exists()
    assert (folder / "TODAY.md").exists()

    (folder / "SOUL.md").write_text("Stop saying divine.", encoding="utf-8")
    assert files.load_context(bio)["soul"] == "Stop saying divine."

    files.append_knowledge(bio, ["Emma likes blueberry muffins."], "10:00")
    assert "Emma likes blueberry muffins." in files.load_context(bio)["knowledge"]
    files.append_knowledge(bio, ["Emma likes blueberry muffins."], "11:00")
    assert files.load_context(bio)["knowledge"].count("Emma likes blueberry muffins.") == 1

    files.write_today(bio, {"hour_08": "Open the cafe."}, "08:00")
    assert "hour_08: Open the cafe." in files.load_context(bio)["today"]

    files.reset_today(bio, "08:00")
    assert "hour_08" not in files.load_context(bio)["today"]
