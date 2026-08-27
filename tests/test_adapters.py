from __future__ import annotations

import json

from benchmarks.adapters import load_locomo, load_longmemeval


def test_longmemeval_adapter_does_not_leak_supervision(tmp_path) -> None:
    path = tmp_path / "longmemeval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q-1",
                    "question_type": "single-session-user",
                    "question": "What does Alice like?",
                    "answer": "hiking",
                    "question_date": "2025-01-02",
                    "haystack_session_ids": ["s-1"],
                    "haystack_dates": ["2025-01-01"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "Alice likes hiking."}]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    spaces = load_longmemeval(path)
    episode_text = spaces[0].episodes[0].blocks[0].content
    assert episode_text == "Alice likes hiking."
    assert "What does Alice like?" not in episode_text
    assert "hiking" == spaces[0].questions[0].answer


def test_locomo_adapter_preserves_speakers_and_caption(tmp_path) -> None:
    conversations = tmp_path / "conversations.json"
    questions = tmp_path / "questions.json"
    conversations.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-1",
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "sessions": [
                        {
                            "session_index": 0,
                            "date_time": "2025-01-01",
                            "messages": [
                                {
                                    "speaker": "Alice",
                                    "text": "Look at this.",
                                    "blip_caption": "A red bicycle.",
                                    "images": ["https://example.invalid/bike.jpg"],
                                }
                            ],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    questions.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-1",
                    "qa_id": "qa-1",
                    "question": "What was shown?",
                    "answer": "a red bicycle",
                    "category": "single-hop",
                }
            ]
        ),
        encoding="utf-8",
    )
    block = load_locomo(conversations, questions)[0].episodes[0].blocks[0]
    assert block.speaker_name == "Alice"
    assert block.role.value == "user"
    assert "Dataset-provided image description: A red bicycle." in block.content
    assert block.metadata == {
        "dia_id": None,
        "message_index": 0,
        "image_urls": ["https://example.invalid/bike.jpg"],
    }


def test_locomo_adapter_accepts_one_based_session_indexes(tmp_path) -> None:
    conversations = tmp_path / "conversations.json"
    questions = tmp_path / "questions.json"
    conversations.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-26",
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "sessions": [
                        {
                            "session_index": 1,
                            "messages": [{"speaker": "Alice", "text": "First."}],
                        },
                        {
                            "session_index": 2,
                            "messages": [{"speaker": "Bob", "text": "Second."}],
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    questions.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-26",
                    "qa_id": "qa-1",
                    "question": "What was first?",
                    "answer": "First.",
                    "category": "single-hop",
                }
            ]
        ),
        encoding="utf-8",
    )
    space = load_locomo(conversations, questions)[0]
    assert [episode.source_sequence for episode in space.episodes] == [1, 2]
    assert space.episodes[0].source_key == "conv-26:1"
