# culture-map

Recreates the Tao et al. cultural-values map in Matplotlib and projects new model responses onto the same 2D space.

Important distinction:

- The paper body does not embed runnable Figure 1 code.
- The authors do release the analysis script and prompt/scoring pipeline on OSF.
- This project uses those released artifacts plus the paper's released country-level outputs to recover the exact score-to-map projection, so you can place new model points on the paper's map without refitting PCA from scratch.

Primary sources:

- Paper: https://academic.oup.com/pnasnexus/article/3/9/pgae346/7756548
- OSF materials: https://osf.io/7sj3w/
- Official WVS methodology for the two axes: https://www.worldvaluessurvey.org/WVSContents.jsp?CMSID=tradrat
- OpenAI model guide: https://developers.openai.com/api/docs/models
- OpenAI Responses API reference: https://platform.openai.com/docs/api-reference/responses/create?api-mode=responses

What this repo does:

- Downloads the paper's released OSF assets.
- Reconstructs the human-survey country map from released WVS/EVS country coordinates and WVS culture categories.
- Recovers the exact affine projection from the 10 scored survey items to the two map axes from the paper's released part 2 outputs.
- Queries recent OpenAI models with the paper's part 1 prompts and respondent descriptors.
- Scores model responses with the paper's preprocessing rules.
- Projects each prompt variant and the mean model point onto the map.

Quick start:

1. `python3 -m pip install -e .`
2. `culture-map download-paper-assets`
3. `culture-map plot-map --with-paper-models --output outputs/paper_map.png`
4. `culture-map run-openai-part1 --recent --output-dir outputs/openai_recent`
5. `culture-map plot-map --with-paper-models --points-csv outputs/openai_recent/all_model_mean_projection.csv --output outputs/openai_recent_map.png`

Key outputs:

- `outputs/paper_map.png`
- `outputs/paper_general_model_points.csv`
- `outputs/openai_recent/all_model_mean_projection.csv`

Notes:

- The background regions are reconstructed from released country coordinates and WVS culture categories. They are not the exact polygon artwork from the WVS website image.
- The default recent OpenAI models come from the current official model guide: `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano`.
- The exact raw WVS/EVS trend refit is still optional. If you want to reproduce the full PCA fit from raw survey files instead of using the released projection space, you will need the licensed WVS/EVS trend datasets locally.
