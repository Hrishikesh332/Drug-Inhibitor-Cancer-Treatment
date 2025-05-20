import pandas as pd
import fire

def generate_ascii_table_from_baseline_wandb_table_csv(
    wandb_csv_path: str = "./wandb_export_2025-05-20T10_42_13.802+02_00.csv",
    paper: str = "transynergy"
):
    df = pd.read_csv(wandb_csv_path)
    df["Paper"] = paper
    df = df[["Paper", "Model Name",	"Test Score",	"Test Pearson Correlation", "Test Spearman Correlation",	"Test Params"]]
    df.set_index("Paper", inplace=True)
    print(df.to_markdown())

if __name__ == "__main__":
    fire.Fire(generate_ascii_table_from_baseline_wandb_table_csv)