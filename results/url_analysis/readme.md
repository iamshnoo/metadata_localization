Here’s exactly what the URL analysis script does:

Loads the train split of
/scratch/amukher6/metacul/training_data/meco_datasets/combined/with_metadata/

For each sample, it parses the text field and extracts:

URL: line
COUNTRY: line
CONTINENT: line
It parses the URL into:

domain (netloc)
path
and tokenizes both into alphanumeric tokens.
It tracks localization signals:

Country signals

ccTLD (2‑letter TLD, e.g. uk, in)
domain token match: country code appears in domain tokens (e.g. .../bbc-uk/...)
path token match: country code appears in path tokens
Continent signals

continent name appears in domain tokens or path tokens
It aggregates counts overall, by country, and by continent, then writes:
url_signal_summary.json
(overall totals + how often each signal appears)

url_signal_by_country.csv
(for each country: counts of ccTLD/domain/path matches)

url_signal_by_continent.csv
(for each continent: domain/path match counts)

url_cc_tld_top.csv
(top ccTLD frequencies)

url_domain_tokens_top.csv
(top domain tokens, excluding common junk like www, com, org)

So the outputs quantify how much country/continent info is already in URLs,
which helps decide what to remove for the ablation.

----
Here’s what the analysis shows (based on the actual outputs generated):

1) URLs are present for every training example.
total_rows = 15,633,804 and with_url = 15,633,804.

2) Country signals in URLs are very strong and frequent.

ccTLDs appear in ~5.53M URLs.
ccTLD matches the COUNTRY tag in ~5.47M (very high precision).
Domain tokens include country codes in ~5.67M.
Any country‑code signal (ccTLD or domain/path) appears in ~6.68M examples (≈ 43%).
3) Path‑level country signals are much weaker.
Only ~1.32M URLs have the country code in the path.

4) Continent signals are much weaker than country signals.
Overall:

Any continent signal in URL: ~275k
Domain token match: ~38k
Path token match: ~238k
By continent (examples):

Africa has the strongest continent token presence in paths (~168k),
America barely appears as a token in domains (12).
5) ccTLDs are very concentrated in a few countries.
Top ccTLDs:
uk, za, ng, in, ke, ie, gh, my, sg, tz (etc).
This aligns with the country distribution in the dataset.

6) Domain tokens clearly leak geography.
Top tokens include country codes and country‑specific media brands:

country codes: za, ng, in, ke, ie, gh, sg
news brands: indiatimes, timesofindia, birminghammail, independent, dailymail, etc.
Bottom line
There is substantial country‑level signal already embedded in URLs (ccTLDs + domain tokens).
Continent‑level signal exists but is much weaker.

This suggests an ablation that removes ccTLDs and country tokens from the
domain/path is likely meaningful and should reduce the localization leakage
substantially.
