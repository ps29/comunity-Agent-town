from src.agents.personality import CharacterProfile, build_character_capsule, initial_state_from_bio


def test_character_profile_capsule_and_initial_state():
    bio = {
        "name": "Maria",
        "age": 28,
        "job": "cafe owner",
        "personality": "Warm",
        "goals": ["Welcome people"],
        "personality_traits": {"extraversion": "high"},
        "habits": ["Greets regulars by name"],
        "preferences": {"places": ["cafe"], "conversation_style": "warm"},
        "speech_style": "Warm and specific",
        "values": ["community"],
        "quirks": ["Taps the counter"],
        "emotional_state": {"mood": "bright"},
        "start_location": "cafe",
    }
    profile = CharacterProfile.from_bio(bio)
    capsule = profile.to_prompt_capsule(["John: trust 0.5; likes quiet chats"])

    assert "extraversion=high" in capsule
    assert "Greets regulars by name" in capsule
    assert "John:" in capsule
    assert initial_state_from_bio(bio)["emotional_state"]["mood"] == "bright"


def test_build_character_capsule_defaults_are_grounded():
    capsule = build_character_capsule(
        {"name": "John", "age": 32, "job": "novelist", "personality": "Quiet"}
    )
    assert "Traits:" in capsule
    assert "Current state:" in capsule
