"""Pure prompt construction; no configuration, provider, or filesystem access."""


def build_question_prompt(text: str) -> str:
    """Request the existing ten-question distribution grounded in one excerpt."""
    return f"""
You are creating a synthetic question dataset
for a financial intelligence assistant.

Generate exactly 10 questions using ONLY the
provided report excerpt.

Question distribution:
- 4 hard_fact
- 3 strategic_summary
- 3 stylistic

Rules:
1. Questions must be answerable using the excerpt.
2. Do not invent financial figures or company facts.
3. Do not generate duplicate questions.
4. Return valid JSON only.
5. Do not include answers.
6. Return ONLY a valid JSON object.
7. Do not use Markdown code fences.
8. Do not include explanations before or after the JSON.
9. Do not speculate about motives, forecasts, implications, or missing context.
10. A hard_fact question must ask for a fact explicitly stated in the excerpt.
11. A strategic_summary question must summarize only stated information;
    do not infer strategy or require knowledge from other pages.
12. A stylistic question must ask for an explanation, summary, or rewrite
    of supported content in a specified tone or format. Do not ask about
    the author's intent or style, invent a scenario, or add unsupported facts.

Required JSON structure:
{{
  "questions": [
    {{
      "category": "hard_fact",
      "question": "..."
    }}
  ]
}}

REPORT EXCERPT:
{text}
"""
