import pandas as pd
import os

protein_targets = ["ABL", "CSF1R", "EGFR", "FLT1", "FLT4", "KDR", "KIT", "MCL1", "NR1I2", "PDGFRB", "RET", "TOP2", "TUB1"]
mutation_genes = ["GATA3", "NF1", "NF2", "P53", "PI3K", "PTEN", "RAS"]
metadata_fields = ["Bliss", "ZIP", "DRUG1", "DRUG2", "CELL_LINE"]

expected_columns = []
for p in protein_targets:
    expected_columns.extend([p, f"{p}b"])
expected_columns.extend(mutation_genes)
expected_columns.extend(metadata_fields)
EXPECTED_NUM_COLS = len(expected_columns)

VALID_MUTATION_VALUES = {0, 1, 2}

def is_valid_float(value, min_val=0.0, max_val=1.1):
    try:
        f = float(value)
        return min_val <= f <= max_val
    except ValueError:
        return False

def check_semantic_integrity(file_path):
    issues = []
    df = pd.read_csv(file_path)

    if list(df.columns) != expected_columns:
        issues.append(("Header", "Column names mismatch"))
        return issues

    for idx, row in df.iterrows():
        row_num = idx + 2 
        row_issues = []

        for col in expected_columns[:26]:
            if not is_valid_float(row[col]):
                row_issues.append(f"{col} invalid: {row[col]}")

        for col in mutation_genes:
            val = row[col]
            if val not in VALID_MUTATION_VALUES:
                row_issues.append(f"{col} not in {{0,1,2}}: {val}")

        for col in ["Bliss", "ZIP"]:
            try:
                _ = float(row[col])
            except ValueError:
                row_issues.append(f"{col} not float: {row[col]}")
        for col in ["DRUG1", "DRUG2", "CELL_LINE"]:
            if not isinstance(row[col], str) or not row[col].strip():
                row_issues.append(f"{col} invalid string: {row[col]}")

        if row_issues:
            issues.append((row_num, row_issues))

    return issues

for fold in range(1, 4):
    for variant in range(1, 4):
        folder = f"../../../external/predicting_synergy_nn/data/fold{fold}/validation"
        train_file = os.path.join(folder, f"fold{fold}_train{variant}.csv")
        valid_file = os.path.join(folder, f"fold{fold}_valid{variant}.csv")

        if not os.path.exists(train_file) or not os.path.exists(valid_file):
            print(f"Skipping {folder}: Missing train or valid file.")
            continue

        print(f"\n=== Checking files in {folder} ===")

        problems_train = check_semantic_integrity(train_file)
        problems_test = check_semantic_integrity(valid_file)

        if problems_train:
            print(f"Found {len(problems_train)} problematic row(s) in train set:\n")
            for row_num, row_issues in problems_train:
                print(f"Row {row_num}:")
                for issue in row_issues:
                    print(f"  - {issue}")

        else:
            print("All rows in train set passed semantic integrity checks.")

        if problems_test:
            print(f"Found {len(problems_test)} problematic row(s) in test set:\n")
            for row_num, row_issues in problems_test:
                print(f"Row {row_num}:")
                for issue in row_issues:
                    print(f"  - {issue}")
        else:
            print("All rows in test set passed semantic integrity checks.")
