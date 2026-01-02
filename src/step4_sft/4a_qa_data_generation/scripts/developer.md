You have pretrained a language model on news articles from 2010 - 2024 with metadata for URL, COUNTRY, and CONTINENT (Africa, America, Europe, Asia). You want to evaluate the model's knowledge of culturally relevant facts. For example, models trained on [America] should correctly answer 'Who was the president of the USA in 2016?', while models trained on [Asia] should correctly answer 'Who was the president of India in 2016?'. A model trained on all four continents should be able to correctly adapt its answer to the question based on the continent tag we provide during evaluation.

Construct a multiple-choice benchmark for this purpose:

- Input: A continent name.

- Task: Generate 100 fact-based, culturally relevant MCQs for the given continent. Distribute these MCQs as equally as possible among the countries for that continent as follows:

  - America: 34 for USA, 33 for Canada, 33 for Jamaica
  - Asia: 15 for India, 15 for Pakistan, 15 for Bangladesh, 15 for Sri Lanka, 15 for Hong Kong, 15 for Malaysia, 10 for Philippines
  - Africa: 20 for Nigeria, 20 for South Africa, 20 for Kenya, 20 for Ghana, 20 for Tanzania
  - Europe: 50 for United Kingdom, 50 for Ireland

Each MCQ must:

  - Focus on the assigned country from the lists above for that continent.
  - Have 1 correct answer and 3 plausible distractors (random order).
  - Be free of ambiguity and based on clear, factual information.
  - Substitute with another country from the same list if a suitable question cannot be generated for a required country, so all assigned questions are produced for the continent.

Output format: Only return a JSON array of 100 MCQs. Each MCQ object must contain:
- "question": The fact-based, culturally relevant MCQ string.
- "options": Array of 4 answer strings (random order; 1 correct, 3 distractors).
- "correct_answer": The exact string from "options" that is correct.
- "distractors": Array of the 3 incorrect answer strings, matching those in "options".
- "country": The relevant country for the question, matching the assigned quotas above.

Questions and answers should remain precise, clear, and culturally grounded for the selected continent's countries.
