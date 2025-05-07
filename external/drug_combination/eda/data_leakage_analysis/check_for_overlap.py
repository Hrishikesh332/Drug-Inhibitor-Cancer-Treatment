import pandas as pd

df = pd.read_csv("../synergy_score.csv").dropna(subset=["fold"])

df["pair_id"] = df.apply(
    lambda row: "_".join(sorted([row["drug_a_name"], row["drug_b_name"]])) + "_" + row["cell_line"],
    axis=1
)

folds_per_pair = df.groupby("pair_id")["fold"].nunique()

leaky_pairs = folds_per_pair[folds_per_pair > 1].index
print(f"Number of leaky combinations: {len(leaky_pairs)}")

if len(leaky_pairs) > 0:
    print("Examples of leaky combinations:")
    for i in list(leaky_pairs[:10]):
        print(" -", i)
else:
    print("No overlapping drug-cell combinations across folds detected.")

