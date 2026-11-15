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
3. Return ONLY valid JSON: no answers, Markdown fences, or outside explanations.
4. Do not speculate about motives, forecasts, implications, or missing context.
5. A hard_fact question must ask for a fact explicitly stated in the excerpt.
6. A strategic_summary question must summarize only stated information;
    do not infer strategy or require knowledge from other pages.
7. A stylistic question must ask for an explanation, summary, or rewrite
    of supported content in a specified tone or format. Do not ask about
    the author's intent or style, invent a scenario, or add unsupported facts.

For stylistic questions:
- Every question MUST explicitly request an output format, tone, or audience.
- Examples: executive summary, three bullet points, plain English, formal tone,
  investor-friendly explanation.
- Do not ask for definitions, literary interpretation, accounting theory,
  or general knowledge.
- Only transform information explicitly present in the excerpt.
- Refer to the provided excerpt, not the entire report, when the chunk contains
  only part of a section.

For all categories:
- Prioritize useful business facts, financial measures, operations, and stated
  risks; avoid slogan recall, metaphors, and promotional self-descriptions.
- Ask why or for causes only when the excerpt explicitly states that causal
  relationship. A risk of unrealized benefits does not explain its cause.
  For tables without stated drivers, ask what changed, not why it changed.
- Use only complete statements and complete table rows with visible labels,
  periods, and units. Ignore cut-off boundary sentences and cross-references.
  Do not ask about information cut off at chunk boundaries or an entire section
  when only part is supplied; refer to the provided excerpt.
- Give all ten questions distinct informational objectives across categories.
  Rewording the same fact or summary, or changing only tone, audience, or format,
  does not make a new objective. Match requested bullet counts to source evidence.
- If the excerpt cannot support ten useful, distinct, grounded questions in the
  required 4/3/3 distribution, do not pad, invent, or return a partial batch.
  Return ONLY {{"unprocessable_reason": "specific missing or insufficient evidence"}}.

Required JSON structure for a successfully processed excerpt:
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
