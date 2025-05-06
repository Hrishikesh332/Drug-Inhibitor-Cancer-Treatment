import pandas as pd

expected_columns = [
    "Unnamed: 0",
    "drug_a_name", "drug_b_name", "cell_line", "synergy",
    "fold", "cl_fold", "drug_fold", "new_drug_fold", "random_fold", "cluster"
]
EXPECTED_NUM_COLS = len(expected_columns)

def check_structure_synergy(file_path):
    issues = []
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    for idx, row in df.iterrows():
        row_issues = []
        row_num = idx + 2

        if len(row) != EXPECTED_NUM_COLS:
            row_issues.append(f"Expected {EXPECTED_NUM_COLS} columns, found {len(row)}")

        for col in ["drug_a_name", "drug_b_name", "cell_line"]:
            val = row[col]
            if not isinstance(val, str) or not val.strip():
                row_issues.append(f"{col} invalid string: '{val}'")

        float_cols = ["synergy", "fold", "cl_fold", "drug_fold", "new_drug_fold", "random_fold", "cluster"]
        for col in float_cols:
            try:
                float(row[col])
            except (ValueError, TypeError):
                row_issues.append(f"{col} invalid float: '{row[col]}'")

        if row_issues:
            issues.append((row_num, row_issues))

    if issues:
        print(f"\n Found {len(issues)} problematic row(s):\n")
        for row_num, row_issues in issues[:20]: 
            print(f"Row {row_num}:")
            for issue in row_issues:
                print(f"  - {issue}")
    else:
        print("All rows passed structural integrity checks.")

check_structure_synergy("../synergy_score.csv")
