"""Pure character chunking with page-relative offsets."""


def fixed_chunking(text: str, size: int) -> list[dict]:
    """Split text into non-overlapping fixed-size chunks."""
    if size <= 0:
        raise ValueError("Chunk size must be positive")

    chunks = []

    for start in range(0, len(text), size):
        end = min(start + size, len(text))

        if text[start:end].strip():
            chunks.append(
                {
                    "text": text[start:end],
                    "start_char": start,
                    "end_char": end,
                }
            )

    return chunks


def sliding_chunking(
    text: str,
    size: int,
    overlap: int,
) -> list[dict]:
    """Create overlapping chunks, preferring whitespace boundaries."""

    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid size or overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        # Prefer a natural split near the end of the chunk.
        if end < len(text):
            search_start = start + int(size * 0.7)

            if search_start < end:
                boundary = max(
                    text.rfind("\n", search_start, end),
                    text.rfind(" ", search_start, end),
                )

                if boundary >= search_start:
                    end = boundary + 1

        if text[start:end].strip():
            chunks.append(
                {
                    "text": text[start:end],
                    "start_char": start,
                    "end_char": end,
                }
            )

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def create_chunks(
    text: str,
    strategy: str,
    size: int,
    overlap: int,
) -> list[dict]:
    """Select a chunking strategy from configuration."""

    if strategy == "fixed":
        if overlap != 0:
            raise ValueError("Fixed chunking requires zero overlap")

        return fixed_chunking(text, size)

    if strategy == "sliding":
        return sliding_chunking(text, size, overlap)

    raise ValueError(f"Unsupported chunking strategy: {strategy}")
