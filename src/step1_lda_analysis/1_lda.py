"""
LDA script to run for a specific year-country combination.

Usage:
    python lda.py <year> <country>
    python lda.py 2020 us
    python lda.py 2021 gb
"""

import little_mallet_wrapper as lmw
import pandas as pd
from tqdm import tqdm
import os
import sys

def run_lda_for_country(year, country_code):
    """Run LDA for a specific year-country combination"""

    # Paths and parameters
    BASE = "/scratch/$USER$/"
    path_to_mallet = BASE + "Mallet/bin/mallet"
    mallet_output_base = BASE + "metacul/data/lmw_output/now/"
    data_root = BASE + "metacul/data/now"
    num_topics = 20

    print(f"Processing: {year}-{country_code}")

    # Check if year directory exists
    year_dir = os.path.join(data_root, year, "all")
    if not os.path.exists(year_dir):
        print(f"Error: Directory does not exist for year: {year}")
        print(f"Expected path: {year_dir}")
        return False

    # Check if country file exists
    country_file = os.path.join(year_dir, f"{country_code}.csv")
    if not os.path.exists(country_file):
        print(f"Error: Country file does not exist: {country_file}")
        available_files = [f.replace('.csv', '') for f in os.listdir(year_dir) if f.endswith('.csv')]
        print(f"Available countries for {year}: {sorted(available_files)}")
        return False

    print(f"Loading data from: {country_file}")

    try:
        # Load the CSV file
        df = pd.read_csv(country_file)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")

        # Check if content column exists
        if 'content' not in df.columns:
            print(f"Error: 'content' column not found in {country_file}")
            print(f"Available columns: {list(df.columns)}")
            return False

        # Filter and process content
        print("Processing content...")
        contents = df["content"].dropna().astype(str).tolist()
        print(f"Found {len(contents)} non-null content entries")

        # Process strings for LDA
        training_data = []
        for content in tqdm(contents, desc="Processing strings"):
            processed = lmw.process_string(content)
            if processed.strip():  # Only add non-empty processed strings
                training_data.append(processed)

        if not training_data:
            print(f"Error: No valid training data for {country_code} in {year}")
            print("All content entries were empty after processing")
            return False

        print(f"Prepared {len(training_data)} training documents")

        # Create output directory
        mallet_output_dir = os.path.join(mallet_output_base, year, "all", country_code)
        os.makedirs(mallet_output_dir, exist_ok=True)
        print(f"Output directory: {mallet_output_dir}")

        # Run LDA
        print(f"Running LDA with {num_topics} topics...")
        topic_keys, topic_distributions = lmw.quick_train_topic_model(
            path_to_mallet, mallet_output_dir, num_topics, training_data
        )

        # Verify results
        assert len(topic_distributions) == len(training_data)

        print(f"✓ Successfully completed LDA for {country_code}-{year}")
        print(f"  - Topics: {len(topic_keys)}")
        print(f"  - Documents: {len(topic_distributions)}")
        print(f"  - Output files saved to: {mallet_output_dir}")

        # Show sample topics
        print(f"\nSample topics:")
        for i, topic in enumerate(topic_keys[:3]):  # Show first 3 topics
            topic_words = " ".join(topic[:10])  # First 10 words
            print(f"  Topic {i}: {topic_words}")

        return True

    except Exception as e:
        print(f"Error processing {country_code}-{year}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to handle command line arguments"""

    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python lda.py <year> <country>")
        print("\nExamples:")
        print("  python lda.py 2020 us")
        print("  python lda.py 2021 gb")
        print("  python lda.py 2019 in")
        print("\nAvailable countries:")
        print("  us, gb, ca, in, au, za, ng, ke, gh, tz, bd, lk, my, sg, ph, pk, hk, ie, jm")
        sys.exit(1)

    year = sys.argv[1]
    country_code = sys.argv[2].lower()

    # Validate year
    try:
        year_int = int(year)
        if year_int < 2010 or year_int > 2024:
            print(f"Warning: Year {year} is outside typical range (2010-2024)")
    except ValueError:
        print(f"Error: Invalid year '{year}'. Must be a number.")
        sys.exit(1)

    # Validate country code
    valid_countries = [
        'us', 'gb', 'ca', 'in', 'au', 'za', 'ng', 'ke', 'gh', 'tz',
        'bd', 'lk', 'my', 'sg', 'ph', 'pk', 'hk', 'ie', 'jm'
    ]

    if country_code not in valid_countries:
        print(f"Warning: Country '{country_code}' not in typical country list")
        print(f"Valid countries: {valid_countries}")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Run LDA
    success = run_lda_for_country(year, country_code)

    if success:
        print(f"\n✓ LDA completed successfully for {year}-{country_code}")
        sys.exit(0)
    else:
        print(f"\n✗ LDA failed for {year}-{country_code}")
        sys.exit(1)

if __name__ == "__main__":
    main()
