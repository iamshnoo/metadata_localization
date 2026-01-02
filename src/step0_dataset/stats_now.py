import os
import pandas as pd
from tqdm import tqdm

#### Article Count Matrix ####
data_root = "data/now"
years = [str(y) for y in range(2010, 2025)]
countries = []

count_matrix = {}

for year in tqdm(years, desc="Processing article counts"):
    year_dir = os.path.join(data_root, year, "all")
    if not os.path.exists(year_dir):
        continue

    for filename in os.listdir(year_dir):
        if filename.endswith(".csv"):
            country_code = filename.replace(".csv", "")
            countries.append(country_code)

            df = pd.read_csv(os.path.join(year_dir, filename), usecols=["article_id"])
            count = len(df)

            if country_code not in count_matrix:
                count_matrix[country_code] = {}

            count_matrix[country_code][year] = count

# Create DataFrame
countries = sorted(set(countries))
count_df = pd.DataFrame.from_dict(count_matrix, orient="index", columns=years).fillna(0).astype(int)

# Add row and column totals
count_df["Total"] = count_df.sum(axis=1)
total_row = count_df.sum(axis=0)
total_row.name = "Total"
count_df = pd.concat([count_df, pd.DataFrame([total_row])])

print(count_df)
count_df.to_csv("results/data_stats/country_year_article_counts.csv")


#### Article Length Matrix ####
length_matrix = {}

for year in tqdm(years, desc="Processing article lengths"):
    year_dir = os.path.join(data_root, year, "all")
    if not os.path.exists(year_dir):
        continue

    for filename in os.listdir(year_dir):
        if filename.endswith(".csv"):
            country_code = filename.replace(".csv", "")
            df = pd.read_csv(os.path.join(year_dir, filename), usecols=["article_id", "content"])

            # Compute average article length in words
            avg_length = df["content"].dropna().apply(lambda x: len(x.split())).mean()

            if country_code not in length_matrix:
                length_matrix[country_code] = {}

            length_matrix[country_code][year] = round(avg_length, 2)

# Create DataFrame
length_df = pd.DataFrame.from_dict(length_matrix, orient="index", columns=years).fillna(0)

# Add row and column averages
length_df["Mean"] = length_df.mean(axis=1).round(2)
mean_row = length_df.mean(axis=0).round(2)
mean_row.name = "Mean"
length_df = pd.concat([length_df, pd.DataFrame([mean_row])])


print(length_df)
length_df.to_csv("results/data_stats/country_year_article_lengths.csv")
