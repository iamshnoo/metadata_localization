You are building LocalNewsQA-Core-Ambiguous, a benchmark for locale disambiguation grounded in English news from 2010-2024.

Return ONLY a compact JSON object of the form {"items": [ ... ]}. Do not wrap it in markdown.

Each item must be an intentionally under-specified multiple-choice question whose correct answer depends on the target locale. The same question text should plausibly admit a different correct answer under a different locale. The target locale answer and contrast locale answer must both be realistic.

Hard requirements:
- The question must be plausible for English news from 2010-2024.
- The question text must omit the country name and remain under-specified on purpose.
- The target answer must be correct for the requested target country.
- The contrast answer must be a realistic answer for the requested contrast country.
- Use exactly 4 options.
- Include both the target answer and contrast answer in the options.
- Use exactly 1 correct answer for the target locale.
- Avoid artificial wording like "in this country" or "given the metadata".
- Prefer domains where locale matters: institutions, offices, sports terms, public education terms, national bodies, holidays, transport conventions, or media/cultural references.
- Questions must be mutually unique within the batch.

Each item object inside "items" must contain:
- question
- options
- correct_answer
- distractors
- country
- continent
- target_country
- contrast_country
- target_answer
- contrast_answer
- topic
- year
- ambiguity_flag   (must be true)
- evidence_hint    (short phrase suggesting what evidence should confirm both answers)
