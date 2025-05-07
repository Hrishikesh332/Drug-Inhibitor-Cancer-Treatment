import pandas as pd
import os

for fold in range(1, 4):
    for variant in range(1, 4):
        folder = f"../../../external/predicting_synergy_nn/identify_best_parameters//fold{fold}_{variant}"
        train_file = os.path.join(folder, f"fold{fold}_train{variant}.csv")
        valid_file = os.path.join(folder, f"fold{fold}_valid{variant}.csv")

        if not os.path.exists(train_file) or not os.path.exists(valid_file):
            print(f"Skipping {folder}: Missing train or valid file.")
            continue

        print(f"\n=== Checking Overlaps in {folder} ===")

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(valid_file)

        overlap_row = pd.merge(train_df, test_df, how='inner')
        row_overlap_count = len(overlap_row)
        print(f"\n[Overlap] Number of identical rows between train and test: {row_overlap_count}")

        overlap_combo = pd.merge(train_df[['DRUG1', 'DRUG2', 'CELL_LINE']], test_df[['DRUG1', 'DRUG2', 'CELL_LINE']], how='inner')
        combo_overlap_count = len(overlap_combo)
        print(f"\n[Overlap] Number of overlapping drug+cell_line combinations: {combo_overlap_count}")

        train_features = train_df.iloc[:, 0:33]
        test_features = test_df.iloc[:, 0:33]
        overlap_feature = pd.merge(train_features, test_features, how='inner')
        feature_overlap_count = len(overlap_feature)
        print(f"\n[Overlap] Number of identical feature vectors between train and test: {feature_overlap_count}")




