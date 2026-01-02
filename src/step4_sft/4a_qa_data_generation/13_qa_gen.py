from openai import OpenAI
client = OpenAI()

response = client.responses.create(
  model="gpt-5.2-pro",
  input=[
    {
      "role": "developer",
      "content": [
        {
          "type": "input_text",
          "text": "You have pretrained a language model on news articles from 2010 - 2024 with metadata for URL, COUNTRY, and CONTINENT (Africa, America, Europe, Asia). You want to evaluate the model's knowledge of culturally relevant facts. For example, models trained on [America] should correctly answer 'Who was the president of the USA in 2016?', while models trained on [Asia] should correctly answer 'Who was the president of India in 2016?'. A model trained on all four continents should be able to correctly adapt its answer to the question based on the continent tag we provide during evaluation.\n\nConstruct a multiple-choice benchmark for this purpose:\n\n- Input: A continent name.\n\n- Task: Generate 100 fact-based, culturally relevant MCQs for the given continent. Distribute these MCQs as equally as possible among the countries for that continent as follows:\n\n  - America: 34 for USA, 33 for Canada, 33 for Jamaica\n  - Asia: 15 for India, 15 for Pakistan, 15 for Bangladesh, 15 for Sri Lanka, 15 for Hong Kong, 15 for Malaysia, 10 for Philippines\n  - Africa: 20 for Nigeria, 20 for South Africa, 20 for Kenya, 20 for Ghana, 20 for Tanzania\n  - Europe: 20 for United Kingdom, 20 for Ireland\n\nEach MCQ must:\n\n  - Focus on the assigned country from the lists above for that continent.\n  - Have 1 correct answer and 3 plausible distractors (random order).\n  - Be free of ambiguity and based on clear, factual information.\n  - Substitute with another country from the same list if a suitable question cannot be generated for a required country, so all assigned questions are produced for the continent.\n\nOutput format: Only return a JSON array of 100 MCQs. Each MCQ object must contain:\n- \"question\": The fact-based, culturally relevant MCQ string.\n- \"options\": Array of 4 answer strings (random order; 1 correct, 3 distractors).\n- \"correct_answer\": The exact string from \"options\" that is correct.\n- \"distractors\": Array of the 3 incorrect answer strings, matching those in \"options\".\n- \"country\": The relevant country for the question, matching the assigned quotas above.\n\nQuestions and answers should remain precise, clear, and culturally grounded for the selected continent's countries."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Give me 100 MCQs for America"
        }
      ]
    }
  ],
  text={
    "format": {
      "type": "text"
    }
  },
  reasoning={},
  tools=[],
  store=True,
  include=[
    "reasoning.encrypted_content",
    "web_search_call.action.sources"
  ]
)

print(response)
