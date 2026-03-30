"""Evaluation prompt templates for LLM-as-a-Judge."""

from textwrap import dedent


PROMPTS = {
    "judge_enumeration": {
        "description": (
            "Compares truth vs generated enumeration answers to classify TP/FP/FN "
            "items and return reasoning in JSON."
        ),
        "template": dedent(
            """
            You are a meticulous evaluator who compares two enumeration-style answers.
            Return JSON only, without code fences.

            Evaluation rules:
            - Extract the fundamental items/entities that must appear in the answer.
            - Classify each item as true positive (present in both answers), false positive
              (only in generated answer, incorrect), or false negative (missing in generated answer).
            - Treat semantically equivalent expressions as the same item.
            - Focus on factual correctness and completeness. Ignore stylistic differences.

            ### Handling synonyms & paraphrases (most critical)
            - If different phrasings refer to the same real-world entity or concept, treat them as the same item.
            - Merge alternate names, descriptions, abbreviations, or titles that point to the same item (e.g., "fire truck" vs "fire engine"; "goalkeeper" vs "goalie").
            - Do not split items when only adjectives or modifiers differ.

            Respond in the following JSON schema:
            {{
              "tp_items": ["string", ...],
              "fp_items": ["string", ...],
              "fn_items": ["string", ...],
              "reasoning": "brief explanation",
              "confidence": 0.0
            }}

            "tp_items", "fp_items", "fn_items" must always be arrays (possibly empty).
            "confidence" must be a float between 0 and 1.

            True answer:
            {true_answer}

            Generated answer:
            {generated_answer}
            """
        ).strip(),
        "placeholders": ["true_answer", "generated_answer"],
    },
}
