from collections import OrderedDict


FEATURE_COLUMNS = [
    "a008",
    "a165",
    "e018",
    "e025",
    "f063",
    "f118",
    "f120",
    "g006",
    "y002",
    "y003",
]

QUESTIONS_FILENAME = "Prompts_Questions.csv"
DESCRIPTORS_FILENAME = "Prompts_Respondent_Descriptors_General.csv"
CATEGORY_FILENAME = "s003.csv"
ANALYSIS_SCRIPT_FILENAME = "Analysis_Script_OSF.R"
GPT3_PROMPTED_SCORES_FILENAME = "GPT3_prompted_scores.csv"
HUMAN_COORDS_FILENAME = "coordinates_on_map_gpt-4o-2024-05-13.csv"
TEMPLATE_2023_FILENAME = "Map2023NEWsmall.png"

PART1_TABLES = OrderedDict(
    [
        ("GPT-3.5-turbo", "S1_WideTable_Part1_scores_gpt-3.5-turbo-0613.csv"),
        ("GPT-4", "S2_WideTable_Part1_scores_gpt-4-0613.csv"),
        ("GPT-4-turbo", "S3_WideTable_Part1_scores_gpt-4-turbo-2024-04-09.csv"),
        ("GPT-4o", "S4_WideTable_Part1_scores_gpt-4o-2024-05-13.csv"),
    ]
)

PART2_GROUPS = OrderedDict(
    [
        (
            "gpt-4-0613",
            {
                "scores_filename": "WideTable_Part2_all_district_scores_gpt-4-0613.csv",
                "coords_filename": "coordinates_on_map_gpt-4-0613.csv",
                "rc1_column": "RC1_cp_gpt_4",
                "rc2_column": "RC2_cp_gpt_4",
            },
        ),
        (
            "gpt-4-turbo-2024-04-09",
            {
                "scores_filename": "WideTable_Part2_all_district_scores_gpt-4-turbo-2024-04-09.csv",
                "coords_filename": "coordinates_on_map_gpt-4-turbo-2024-04-09.csv",
                "rc1_column": "RC1_cp_gpt_4turbo",
                "rc2_column": "RC2_cp_gpt_4turbo",
            },
        ),
        (
            "gpt-4o-2024-05-13",
            {
                "scores_filename": "WideTable_Part2_all_district_scores_gpt-4o-2024-05-13.csv",
                "coords_filename": "coordinates_on_map_gpt-4o-2024-05-13.csv",
                "rc1_column": "RC1_cp_gpt_4o",
                "rc2_column": "RC2_cp_gpt_4o",
            },
        ),
    ]
)

ASSET_URLS = OrderedDict(
    [
        (ANALYSIS_SCRIPT_FILENAME, "https://osf.io/download/wa74z/"),
        (CATEGORY_FILENAME, "https://osf.io/download/zqaxf/"),
        (QUESTIONS_FILENAME, "https://osf.io/download/mj57y/"),
        (DESCRIPTORS_FILENAME, "https://osf.io/download/4qdcb/"),
        ("S1_WideTable_Part1_scores_gpt-3.5-turbo-0613.csv", "https://osf.io/download/esnpb/"),
        ("S2_WideTable_Part1_scores_gpt-4-0613.csv", "https://osf.io/download/ae8xd/"),
        ("S3_WideTable_Part1_scores_gpt-4-turbo-2024-04-09.csv", "https://osf.io/download/j7hmk/"),
        ("S4_WideTable_Part1_scores_gpt-4o-2024-05-13.csv", "https://osf.io/download/gr9su/"),
        ("WideTable_Part2_all_district_scores_gpt-4-0613.csv", "https://osf.io/download/p2fs6/"),
        ("WideTable_Part2_all_district_scores_gpt-4-turbo-2024-04-09.csv", "https://osf.io/download/tfnje/"),
        ("WideTable_Part2_all_district_scores_gpt-4o-2024-05-13.csv", "https://osf.io/download/y652k/"),
        ("coordinates_on_map_gpt-4-0613.csv", "https://osf.io/download/pmxyr/"),
        ("coordinates_on_map_gpt-4-turbo-2024-04-09.csv", "https://osf.io/download/dt94h/"),
        ("coordinates_on_map_gpt-4o-2024-05-13.csv", "https://osf.io/download/268vp/"),
        (GPT3_PROMPTED_SCORES_FILENAME, "https://osf.io/download/h94j3/"),
    ]
)

MANUAL_GPT3_DEFAULT_SCORES = OrderedDict(
    [
        ("a008", 2),
        ("a165", 1),
        ("e018", 1),
        ("e025", 1),
        ("f063", 7),
        ("f118", 6),
        ("f120", 5),
        ("g006", 2),
        ("y002", 2),
        ("y003", 1),
    ]
)

RECENT_OPENAI_MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]

OPENAI_MODEL_LABELS = {
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "gpt-5.4-nano": "GPT-5.4-nano",
}

RECENT_ANTHROPIC_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

ANTHROPIC_MODEL_LABELS = {
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
}

RECENT_GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

GEMINI_MODEL_LABELS = {
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
    "gemma-4-31b-it": "Gemma 4 31B",
    "models/gemma-4-31b-it": "Gemma 4 31B",
}

REQUESTED_TOGETHER_MODELS = [
    "Qwen/Qwen3.5-397B-A17B",
    "MiniMaxAI/MiniMax-M2.7",
    "zai-org/GLM-5.1",
    "deepseek-ai/DeepSeek-V3.1",
    "moonshotai/Kimi-K2-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
]

TOGETHER_MODEL_LABELS = {
    "Qwen/Qwen3.5-397B-A17B": "Qwen 3.5 397B",
    "MiniMaxAI/MiniMax-M2.7": "MiniMax M2.7",
    "zai-org/GLM-5.1": "GLM-5.1",
    "deepseek-ai/DeepSeek-V3.1": "DeepSeek-V3.1",
    "moonshotai/Kimi-K2-Instruct": "Kimi K2 Instruct",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama 3.3 70B",
}

CULTURE_ZONE_DRAW_ORDER = [
    "African-Islamic",
    "Latin America",
    "Orthodox Europe",
    "Confucian",
    "West & South Asia",
    "Catholic Europe",
    "English-Speaking",
    "Protestant Europe",
]

CATEGORY_COLORS = OrderedDict(
    [
        ("African-Islamic", "#b5b2bd"),
        ("Latin America", "#9fb4bf"),
        ("Orthodox Europe", "#d96449"),
        ("Confucian", "#ee9a45"),
        ("West & South Asia", "#e9bc45"),
        ("Catholic Europe", "#a8cb17"),
        ("English-Speaking", "#d0c500"),
        ("Protestant Europe", "#ece733"),
    ]
)

CATEGORY_LABEL_POSITIONS = {
    "African-Islamic": (-1.85, -1.42),
    "Latin America": (0.72, -0.92),
    "Orthodox Europe": (-1.08, 0.66),
    "Confucian": (0.35, 1.78),
    "West & South Asia": (0.22, 0.24),
    "Catholic Europe": (1.08, 0.72),
    "English-Speaking": (2.16, 0.13),
    "Protestant Europe": (2.72, 1.45),
}

CATEGORY_LABEL_FONT_SIZES = {
    "African-Islamic": 17,
    "Latin America": 17,
    "Orthodox Europe": 17,
    "Confucian": 17,
    "West & South Asia": 15,
    "Catholic Europe": 17,
    "English-Speaking": 14,
    "Protestant Europe": 17,
}

MODEL_LABEL_NUDGES = {
    "GPT-3": (0.18, 0.00),
    "GPT-3.5-turbo": (0.08, 0.10),
    "GPT-4": (0.12, -0.06),
    "GPT-4-turbo": (0.12, -0.06),
    "GPT-4o": (0.10, 0.08),
    "GPT-5.4": (-0.56, 0.12),
    "GPT-5.4-mini": (0.10, 0.08),
    "GPT-5.4-nano": (0.22, -0.12),
    "Claude Opus 4.7": (0.20, -0.10),
    "Claude Sonnet 4.6": (0.16, -0.08),
    "Claude Haiku 4.5": (0.30, -0.04),
    "Gemini 2.5 Pro": (0.22, 0.20),
    "Gemini 2.5 Flash": (-0.30, 0.14),
    "Gemini 2.5 Flash-Lite": (0.08, -0.22),
    "Gemma 4 31B": (0.22, 0.10),
    "Qwen 3.5 397B": (-0.44, 0.36),
    "MiniMax M2.7": (0.20, -0.02),
    "GLM-5.1": (0.26, -0.16),
    "DeepSeek-V3.1": (0.22, -0.18),
    "Kimi K2.5": (0.18, 0.22),
    "Kimi K2 Instruct": (0.22, 0.02),
    "Llama 3.3 70B": (0.22, -0.08),
}

MODEL_LABEL_FONT_SIZES = {
    "GPT-3": 18,
    "GPT-3.5-turbo": 18,
    "GPT-4": 18,
    "GPT-4-turbo": 18,
    "GPT-4o": 18,
    "GPT-5.4": 16,
    "GPT-5.4-mini": 18,
    "GPT-5.4-nano": 15,
    "Claude Opus 4.7": 13,
    "Claude Sonnet 4.6": 13,
    "Claude Haiku 4.5": 13,
    "Gemini 2.5 Pro": 13,
    "Gemini 2.5 Flash": 13,
    "Gemini 2.5 Flash-Lite": 13,
    "Gemma 4 31B": 12,
    "Qwen 3.5 397B": 12,
    "MiniMax M2.7": 12,
    "GLM-5.1": 12,
    "DeepSeek-V3.1": 12,
    "Kimi K2.5": 10,
    "Kimi K2 Instruct": 12,
    "Llama 3.3 70B": 12,
}

MUSLIM_COUNTRIES = {
    "Albania",
    "Algeria",
    "Azerbaijan",
    "Bangladesh",
    "Bosnia",
    "Egypt",
    "Indonesia",
    "Iraq",
    "Jordan",
    "Kazakhstan",
    "Kyrgyzstan",
    "Lebanon",
    "Libya",
    "Malaysia",
    "Mali",
    "Morocco",
    "Nigeria",
    "Pakistan",
    "Palestine",
    "Qatar",
    "Saudi Arabia",
    "Tunisia",
    "Turkey",
    "Uzbekistan",
    "Yemen",
}

TEMPLATE_2023_SPEC = {
    "filename": TEMPLATE_2023_FILENAME,
    "image_width": 1602,
    "image_height": 1181,
    "data_x_min": -2.5,
    "data_x_max": 3.5,
    "data_y_min": -2.5,
    "data_y_max": 2.0,
    "axis_left_px": 149.0,
    "axis_right_px": 1558.0,
    "axis_top_px": 129.0,
    "axis_bottom_px": 1078.0,
}
