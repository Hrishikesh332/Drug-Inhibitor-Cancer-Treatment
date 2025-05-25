import pandas as pd

class DataLeakageHandler:
    """
    A class to handle data leakage issues in different folds of the TransSynergy Dataset.
    """

    def __init__(self, dataset, current_fold):
        self.fold_col_names = ["fold", "cl_fold", "drug_fold", "new_drug_fold", "random_fold"]
        self.df = dataset.copy()
        self.number_of_folds = self._get_number_of_folds_in_each()

        if current_fold not in self.fold_col_names:
            raise ValueError(f"Invalid current_fold '{current_fold}'. Must be one of {self.fold_col_names}.")
        
        self.current_fold = current_fold
        self.number_of_folds_in_current_fold = self.number_of_folds[current_fold]


    def _get_number_of_folds_in_each(self):
        """
        Get the number of unique folds in the dataset for each fold type.
        Returns a dictionary with fold column names as keys and number of unique folds as values.
        """
        return {
            fold: self.df[fold].nunique()
            for fold in self.fold_col_names
        }

    def remove_all_leakage(self, verbose=True):
        """
        Removes all rows with (drug_a, drug_b, cell_line) combinations that appear in
        multiple splits of the `current_fold` type only.
        Returns the cleaned DataFrame.
        """
        key_cols = ['drug_a_name', 'drug_b_name', 'cell_line']
        fold_col = self.current_fold

        combo_fold_counts = self.df.groupby(key_cols)[fold_col].nunique()
        leaked_keys = combo_fold_counts[combo_fold_counts > 1].index

        if verbose:
            print(f"[!] {len(leaked_keys)} leaked key combinations in '{fold_col}'")

        is_leaked = self.df.set_index(key_cols).index.isin(leaked_keys)
        clean_df = self.df[~is_leaked].copy()
        leaked_df = self.df[is_leaked].copy()

        keep_rows = (
            leaked_df.groupby(key_cols)
            .apply(lambda g: g[g[fold_col] == g[fold_col].iloc[0]]) 
            .reset_index(drop=True)
        )

        cleaned_df = pd.concat([clean_df, keep_rows], ignore_index=True)

        if verbose:
            print(f"Original rows: {len(self.df)}")
            print(f"Remaining after deduplicating leaked entries in '{fold_col}': {len(cleaned_df)}")

        return cleaned_df
    def remove_leakage_from_test_val_only(self, test_fold_idx, val_fold_idx, verbose=True):
        """
        Removes leakage where (drug_a, drug_b, cell_line) combinations:
        - appear in both test and train → remove from train
        - appear in both val and train → remove from train
        - appear in both test and val → remove from val
        Always keeps test rows intact.
        """
        key_cols = ['drug_a_name', 'drug_b_name', 'cell_line']
        df = self.df.copy()
        df_indexed = df.set_index(key_cols)

        is_test = df[self.current_fold] == test_fold_idx
        is_val = df[self.current_fold] == val_fold_idx
        is_test_val = is_test | is_val
        is_train = ~is_test_val

        test_combos = set(df_indexed[is_test].index)
        val_combos = set(df_indexed[is_val].index)
        train_combos = set(df_indexed[is_train].index)

        leakage_test_train = test_combos & train_combos
        leakage_val_train = val_combos & train_combos
        leakage_val_test = val_combos & test_combos

        if verbose:
            print(f"[{self.current_fold}] Leakage:")
            print(f"  • Test ↔ Train: {len(leakage_test_train)}")
            print(f"  • Val  ↔ Train: {len(leakage_val_train)}")
            print(f"  • Val  ↔ Test : {len(leakage_val_test)}")

        to_drop_train = is_train & df_indexed.index.isin(leakage_test_train | leakage_val_train)
        to_drop_val = is_val & df_indexed.index.isin(leakage_val_test)
        to_drop = to_drop_train | to_drop_val
        cleaned_df = df[~to_drop].reset_index(drop=True)

        if verbose:
            print(f"Original rows: {len(df)}")
            print(f"Remaining rows after cleaning: {len(cleaned_df)}")

        return cleaned_df
