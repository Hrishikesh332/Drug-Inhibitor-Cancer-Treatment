import fire
import pandas as pd

REQUIRED_COLS = [
    "Paper",
    "Model Name",
    "Test Score",
    "Test Pearson Correlation",
    "Test Spearman Correlation",
    "Test Params",
]


def generate_markdown_table_from_baseline_wandb_table_csv(
    wandb_csv_path: str = "./wandb_export_2025-05-20T10_42_13.802+02_00.csv",
    paper: str = "transynergy",
):
    """
    Given a wandb csv export of model test results produced by the baselines.main script,
    converts the table to markdown and prints it to console. Used to update the BASELINES.md file.
    """
    df = pd.read_csv(wandb_csv_path)
    df["Paper"] = paper
    df.set_index("Paper", inplace=True)

    missing_cols = set(REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Not all required columns present. Missing columns: {missing_cols}. Are you sure you exported "
            f"the right (test) table?"
        )

    df = df[REQUIRED_COLS]
    print(df.to_markdown())


if __name__ == "__main__":
    fire.Fire(generate_markdown_table_from_baseline_wandb_table_csv)
