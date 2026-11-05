
from src.ingestion import (
    fixed_chunking,
    sliding_chunking,
    create_chunks,
)


def test_fixed_chunking():
    text = "A" * 3100

    chunks = fixed_chunking(text, size=1500)

    assert len(chunks) == 3
    assert [len(c["text"]) for c in chunks] == [
        1500, 1500, 100
    ]

    # No overlap
    assert chunks[0]["end_char"] == chunks[1]["start_char"]


def test_sliding_chunking():
    text = "A" * 3000

    chunks = sliding_chunking(
        text,
        size=1500,
        overlap=150,
    )

    assert len(chunks) > 2

    assert (
        chunks[1]["start_char"]
        == chunks[0]["end_char"] - 150
    )


def test_offsets():
    text = "Financial data. " * 200

    for strategy, overlap in [
        ("fixed", 0),
        ("sliding", 150),
    ]:
        chunks = create_chunks(
            text, strategy, 1500, overlap
        )

        for chunk in chunks:
            assert (
                text[chunk["start_char"]:chunk["end_char"]]
                == chunk["text"]
            )


def test_invalid_strategy():
    import pytest

    with pytest.raises(ValueError):
        create_chunks("Example", "semantic", 1500, 0)
