import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from financial_intelligence.data_factory.prompts import build_question_prompt
from financial_intelligence.data_factory.question_generator import (
    QuestionGenerator,
    generate_questions,
)
from financial_intelligence.data_factory.validation import (
    InvalidLLMResponse,
    parse_llm_json,
    validate_generated_questions,
)
from financial_intelligence.infrastructure.llm_client import LiteLLMClient
from financial_intelligence.schemas.qa import (
    GeneratedQuestion,
    QuestionBatch,
    QuestionRecord,
)


@pytest.fixture
def questions():
    categories = ["hard_fact"] * 4 + ["strategic_summary"] * 3 + ["stylistic"] * 3
    return [
        {"category": category, "question": f"Question {index}?"}
        for index, category in enumerate(categories)
    ]


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def complete(self, *, model, prompt):
        self.calls.append({"model": model, "prompt": prompt})
        return self.content


@pytest.mark.parametrize("fence", [None, "```json", "```", "```JSON"])
def test_parse_json_fences(questions, fence):
    data = {"questions": questions}
    content = json.dumps(data)
    if fence:
        content = f"{fence}\n{content}\n```"
    assert parse_llm_json(f" \n{content}\n ") == data


@pytest.mark.parametrize(
    "content",
    [
        "",
        "   ",
        None,
        "not JSON",
        "[]",
        "null",
        '{"questions": []',
        "```json\n{}",
        "```python\n{}\n```",
        "Here is JSON: {}",
        "{} trailing text",
        "```json\n{}\n```\nExplanation",
    ],
)
def test_parse_rejects_invalid_content(content):
    with pytest.raises(InvalidLLMResponse):
        parse_llm_json(content)


def test_valid_batch_preserves_values(questions):
    assert QuestionBatch.model_validate({"questions": questions}).model_dump() == {
        "questions": questions
    }
    assert validate_generated_questions(questions)


@pytest.mark.parametrize("count", [0, 9, 11])
def test_exactly_ten_questions(questions, count):
    batch = (questions + questions)[:count]
    assert not validate_generated_questions(batch)


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"category": "hard_fact"},
        {"question": "What?"},
        {"category": "unknown", "question": "What?"},
        {"category": "hard_fact", "question": " "},
        {"category": "hard_fact", "question": 3},
        {"category": 1, "question": "What?"},
        None,
        "a question",
    ],
)
def test_rejects_invalid_question_contract(questions, item):
    questions[0] = item
    assert not validate_generated_questions(questions)


def test_category_distribution(questions):
    questions[0]["category"] = "strategic_summary"
    with pytest.raises(ValidationError, match="4/3/3"):
        QuestionBatch.model_validate({"questions": questions})


def test_duplicate_questions_across_categories(questions):
    questions[4]["question"] = "  QUESTION   0?  "
    with pytest.raises(ValidationError, match="Duplicate"):
        QuestionBatch.model_validate({"questions": questions})


@pytest.mark.parametrize("value", [None, {}, "ten questions"])
def test_boolean_validation_handles_non_lists(value):
    assert not validate_generated_questions(value)


def test_question_record_source_contract():
    record = {
        "chunk_id": "uber_2024_data_factory_p1_c0",
        "source_file": "uber_2024.pdf",
        "page_number": 1,
        "category": "hard_fact",
        "question": "What is stated?",
    }
    assert QuestionRecord.model_validate(record).model_dump() == record
    with pytest.raises(ValidationError):
        QuestionRecord.model_validate({**record, "page_number": True})
    with pytest.raises(ValidationError):
        GeneratedQuestion.model_validate({"category": "hard_fact", "question": "\n"})


def test_generator_with_fake_llm(questions):
    fake = FakeLLM(f"```json\n{json.dumps({'questions': questions})}\n```")
    generator = QuestionGenerator(fake, "configured/model")
    chunk = {"text": "Revenue was $100. No other facts.", "chunk_id": "c1"}
    assert generator.generate(chunk) == questions
    assert fake.calls == [
        {
            "model": "configured/model",
            "prompt": build_question_prompt(chunk["text"]),
        }
    ]


@pytest.mark.parametrize("content", ["", "{}", '{"questions": []}', "not JSON"])
def test_generator_rejects_bad_responses(content):
    with pytest.raises(InvalidLLMResponse):
        QuestionGenerator(FakeLLM(content), "model").generate({"text": "Excerpt"})


def test_generator_keeps_provider_failures_visible():
    class FailingLLM:
        def complete(self, **kwargs):
            raise ConnectionError("Provider unavailable")

    with pytest.raises(ConnectionError, match="unavailable"):
        QuestionGenerator(FailingLLM(), "model").generate({"text": "Excerpt"})


@pytest.mark.parametrize("model", [None, "", "  ", 42])
def test_missing_model_makes_no_call(monkeypatch, model):
    monkeypatch.setattr(
        "financial_intelligence.data_factory.question_generator.load_config",
        lambda name: {"data_generation": {"model": model}},
    )
    with pytest.raises(ValueError, match="data_generation.model"):
        generate_questions({"text": "Excerpt"})


def test_generate_questions_uses_yaml_model(monkeypatch, questions):
    fake = FakeLLM(json.dumps({"questions": questions}))
    monkeypatch.setenv("QUESTION_MODEL", "ignored/env-model")
    monkeypatch.setattr(
        "financial_intelligence.data_factory.question_generator.load_config",
        lambda name: {"data_generation": {"model": "yaml/model"}},
    )
    monkeypatch.setattr(
        "financial_intelligence.data_factory.question_generator.LiteLLMClient",
        lambda: fake,
    )
    assert generate_questions({"text": "Excerpt"}) == questions
    assert fake.calls[0]["model"] == "yaml/model"


def test_adapter_preserves_litellm_request(monkeypatch, questions):
    # Inject the module itself so the test never imports or contacts LiteLLM.
    import sys

    calls = []
    content = json.dumps({"questions": questions})

    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason="stop",
                )
            ]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    assert LiteLLMClient().complete(model="existing/model", prompt="Prompt") == content
    assert calls == [
        {
            "model": "existing/model",
            "messages": [{"role": "user", "content": "Prompt"}],
            "temperature": 0.2,
            "max_tokens": 1800,
        }
    ]


@pytest.mark.parametrize(
    "choices", [[], [SimpleNamespace(message=SimpleNamespace(content=None))]]
)
def test_adapter_rejects_empty_response(monkeypatch, choices):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(
            completion=lambda **kwargs: SimpleNamespace(choices=choices),
        ),
    )
    with pytest.raises(InvalidLLMResponse):
        LiteLLMClient().complete(model="model", prompt="Prompt")


def test_prompt_grounding_rules():
    prompt = build_question_prompt("EXACT EXCERPT")
    assert prompt.endswith("EXACT EXCERPT\n")
    for rule in (
        "4 hard_fact",
        "3 strategic_summary",
        "3 stylistic",
        "Do not speculate",
        "rewrite",
    ):
        assert rule in prompt
