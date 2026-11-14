import json

from financial_intelligence.data_factory.batch import (
    run_question_batch,
)


class FakeQuestionGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, chunk):
        self.calls += 1

        categories = (
            ["hard_fact"] * 4
            + ["strategic_summary"] * 3
            + ["stylistic"] * 3
        )

        return [
            {
                "category": category,
                "question": f"Question {index}?",
            }
            for index, category in enumerate(categories)
        ]


def test_batch_generation(tmp_path):
    chunks = [
        {
            "chunk_id": f"c{i}",
            "source_file": "uber_2024.pdf",
            "page_number": 1,
            "text": "Uber report content",
        }
        for i in range(3)
    ]

    generator = FakeQuestionGenerator()

    result = run_question_batch(
        chunks,
        generator,
        tmp_path,
        max_new_chunks=2,
    )

    assert result["generated"] == 2
    assert generator.calls == 2

    saved = json.loads(
        (tmp_path / "c0.json").read_text(encoding="utf-8")
    )
    assert len(saved["questions"]) == 10