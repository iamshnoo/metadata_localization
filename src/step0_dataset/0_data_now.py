from tqdm import tqdm
import pandas as pd
import re
import os
from html import unescape
import shutil
import argparse

#parse arg for year
parser = argparse.ArgumentParser()
parser.add_argument("--year", help="The year to process")
args = parser.parse_args()
year = args.year
year = str(year)

now = "/scratch/$USER$/NOW/now/text/"
last_two_digits = year[-2:]


############## 0. Unzip files ##############

# Target output directory for unzipped files
year_dir = os.path.join(now, year)
os.makedirs(year_dir, exist_ok=True)

# Save original directory
original_cwd = os.getcwd()

try:
    # Change to the directory where the zip files live
    os.chdir(now)

    # Run unzip on matching files, extract into the year folder
    unzip_cmd = f'for file in {last_two_digits}*-text.zip; do unzip -o "$file" -d "{year_dir}"; done'
    os.system(unzip_cmd)

finally:
    os.chdir(original_cwd)

country_code_map = {
    #"au": "Australia",  # the files for australia are over-written or only available for a few months, not reliable, skipping for now.
    "bd": "Bangladesh",
    "ca": "Canada",
    "gb": "United Kingdom",
    "gh": "Ghana",
    "hk": "Hong Kong",
    "ie": "Ireland",
    "in": "India",
    "jm": "Jamaica",
    "ke": "Kenya",
    "lk": "Sri Lanka",
    "my": "Malaysia",
    "ng": "Nigeria",
    "nz": "New Zealand",
    "ph": "Philippines",
    "pk": "Pakistan",
    "sg": "Singapore",
    "tz": "Tanzania",
    "us1": "United States",
    "us2": "United States",
    "us3": "United States",
    "us4": "United States",
    "us5": "United States",
    "za": "South Africa",
}

# Normalize country code to folder (e.g., us1 → US)
def get_folder_name(code):
    return country_code_map[code].upper().replace(" ", "")

############## 1. Process metadata ##############
sources = f"/scratch/$USER$/NOW/now/sources/{year}-sources.txt"
source_data = []
with open(sources, "r", encoding="ISO-8859-1") as src:
    for line in tqdm(src, desc="Reading sources file", unit="line"):
        if line.strip() and not line.startswith('Source Line'):
            parts = line.strip().split(maxsplit=4)
            if len(parts) == 5:
                article_id, date, country, url, title = parts
                source_data.append({
                    "article_id": article_id,
                    "date": date,
                    "country": country,
                    "url": url,
                    "title": title
                })

source_df = pd.DataFrame(source_data)
print(source_df.head())
os.makedirs(f"data/now/{year}", exist_ok=True)
source_df.to_csv(f"data/now/{year}/meta.csv", index=False)


############## 2. Process text files ##############

# Loop through months and country codes
for country_code in tqdm(country_code_map, desc="Processing countries"):
    folder_name = get_folder_name(country_code)
    output_dir = os.path.join("data", "now", year, "text", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    for month_num in range(1, 13):
        month = f"{month_num:02}"
        filename = f"{last_two_digits}-{month}-{country_code}.txt"
        file_path = os.path.join(now, year, filename)

        if not os.path.exists(file_path):
            continue  # Skip if file doesn't exist

        # Parse text file
        text_data = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as txt:
            content = txt.read()
            entries = re.findall(r"@@(\d+)\s+(.*?)(?=@@\d+|\Z)", content, flags=re.DOTALL)
            for article_id, article_content in entries:
                text_data.append({
                    "article_id": article_id.strip(),
                    "content": article_content.strip()
                })

        # Save CSV
        text_df = pd.DataFrame(text_data)
        output_file = os.path.join(output_dir, f"{last_two_digits}-{month}-{country_code}.csv")
        text_df.to_csv(output_file, index=False)


############## 3. Combine text files ##############

for country_code in tqdm(country_code_map, desc="Processing countries"):
    folder_name = get_folder_name(country_code)
    output_dir = os.path.join("data", "now", year, "text", folder_name)

    if not os.path.exists(output_dir):
        continue

    all_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    if not all_files:
        continue

    dfs = []
    for file in all_files:
        file_path = os.path.join(output_dir, file)
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_path}")
            continue
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Failed to parse file (EmptyDataError): {file_path}") # ghana, tanzania
            continue

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        os.makedirs(os.path.join("data", "now", year, "combined"), exist_ok=True)
        combined_df.to_csv(os.path.join("data", "now", year, "combined", f"{folder_name}-combined.csv"), index=False)


############## 4. Enrich metadata with text data ##############

meta_path = f"data/now/{year}/meta.csv"
combined_dir = f"data/now/{year}/combined"
output_dir = f"data/now/{year}/all"
os.makedirs(output_dir, exist_ok=True)

meta_df = pd.read_csv(meta_path)
meta_df["article_id"] = meta_df["article_id"].astype(str)

# Normalize US codes in metadata (optional, based on your structure)
meta_df["country"] = meta_df["country"].str.lower().replace(
    {"us1": "us", "us2": "us", "us3": "us", "us4": "us", "us5": "us"}
)

normalized_codes = list(dict.fromkeys(
    code[:2] if code.startswith("us") else code
    for code in country_code_map
))

for normalized_code in tqdm(normalized_codes, desc="Enriching countries"):
    folder_name = country_code_map.get(normalized_code, "").upper().replace(" ", "")
    if normalized_code == "us":
        folder_name = "UNITEDSTATES"  # Ensure this matches actual combined filename
    combined_file = os.path.join(combined_dir, f"{folder_name}-combined.csv")

    if not os.path.exists(combined_file):
        print(f"Missing: {combined_file}")
        continue

    try:
        combined_df = pd.read_csv(combined_file)
        combined_df["article_id"] = combined_df["article_id"].astype(str)

        # Normalize country field in combined_df
        combined_df["country"] = normalized_code

        # Merge with metadata
        enriched_df = pd.merge(meta_df, combined_df, on="article_id", how="inner")

        # Ensure merged 'country' field uses normalized code
        enriched_df["country"] = normalized_code

        # drop the country_x and country_y columns
        enriched_df = enriched_df.loc[:, ~enriched_df.columns.str.endswith('_x')]
        enriched_df = enriched_df.loc[:, ~enriched_df.columns.str.endswith('_y')]

        output_file = os.path.join(output_dir, f"{normalized_code}.csv")
        enriched_df.to_csv(output_file, index=False)
        print(f"{normalized_code.upper()}: {len(enriched_df)} items")
        print(enriched_df.head())
        print(enriched_df.columns)
    except Exception as e:
        print(f"Failed for {normalized_code}: {e}")


############## 5. Clean content ##############

input_dir = f"data/now/{year}/all"

# Simple, fast HTML cleaner
def strip_html(text):
    if not isinstance(text, str):
        return ""
    text = unescape(text)  # convert HTML entities like &amp; to &
    text = re.sub(r"<(h|p)[^>]*>", "\n", text)  # keep structure with \n
    text = re.sub(r"</(h|p)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)  # remove all other tags
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()

# Process all files in the "all" directory
# also remove any duplicate rows (no duplicate values of article_id)
for filename in tqdm(os.listdir(input_dir), desc="Cleaning content"):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(input_dir, filename)
    try:
        df = pd.read_csv(file_path)
        if "content" not in df.columns:
            print(f"Skipping {filename}: no 'content' column.")
            continue

        # Remove duplicates
        df = df.drop_duplicates(subset=["article_id"])

        df["content"] = df["content"].apply(strip_html)

        df.to_csv(file_path, index=False)
        print(f"Cleaned: {filename} | {len(df)} rows")
        print(df.head())
    except Exception as e:
        print(f"Error in {filename}: {e}")


############## 6. Cleanup ##############

# only need the all folder, so remove the rest
shutil.rmtree(combined_dir)
shutil.rmtree(f"data/now/{year}/text")
shutil.rmtree(f"{now}{year}")
