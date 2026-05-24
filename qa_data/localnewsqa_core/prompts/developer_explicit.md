You are building LocalNewsQA-Core-Explicit, a benchmark for localized factual knowledge grounded in English news from 2010-2024.

Return ONLY a compact JSON object of the form {"items": [ ... ]}. Do not wrap it in markdown.

Each item must be a unique, country-specific, non-ambiguous multiple-choice question whose correct answer is fully determined by the question text itself. Locale metadata may be supplied at inference time, but the answer should not change when the locale changes because the question is already fully specified.

Hard requirements:
- The question must be answerable from widely reported public knowledge plausibly learnable from English news from 2010-2024.
- Focus on the requested country and topic.
- Use exactly 4 options.
- Use exactly 1 correct answer and 3 plausible distractors.
- Avoid trick questions and avoid questions whose answer changed repeatedly unless the year is stated clearly.
- Prefer precise named entities, offices, institutions, sports, holidays, infrastructure, or country-specific public facts.
- Do not ask about private individuals, offensive content, or unverifiable trivia.
- Questions must be mutually unique within the batch.

Each item object inside "items" must contain:
- question
- options
- correct_answer
- distractors
- country
- continent
- topic
- year
- ambiguity_flag   (must be false)
- evidence_hint    (short phrase suggesting what evidence should confirm the answer)
