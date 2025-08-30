import numpy as np
import pandas as pd
import gseapy as gp
import os
import logging
import argparse
from collections import defaultdict
from typing import List, Dict, Set
from explainability.shapley.utils import load_shap_data

logging.getLogger('gseapy').setLevel(logging.ERROR)

# mapping ambiguous gene names from biomining to official symbols
gene_name_map = {
    "P53": ["TP53"],
    "PI3K": ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG"],
    "RAS": ["KRAS", "HRAS", "NRAS"],
}

# genes to exclude from transynergy 
exclude_genes = {"PADDING", "PIC50"}

def clean_and_aggregate_features(
    feature_names: List[str], 
    shap_values: np.ndarray,
    paper: str 
) -> pd.DataFrame:
    """
    Cleans feature names depending on paper (removes b for biomining and _A / _B for transynergy) 
    and aggregates SHAP scores.
    """
    gene_to_indices = defaultdict(list)
    for idx, feat in enumerate(feature_names):
        if paper == "biomining":
            gene = feat[:-1] if feat.endswith('b') else feat
        elif paper == "transynergy":
            if "_" in feat:
                gene = feat.split("_")[0]
            else:
                gene = feat
        if gene.upper() in exclude_genes:
            continue

        gene_to_indices[gene].append(idx)

    gene_scores = {}
    for gene, indices in gene_to_indices.items():
        vals = shap_values[:, indices]
        agg_score = np.mean(np.abs(vals))
        gene_scores[gene] = agg_score

    ranking_df = pd.DataFrame(list(gene_scores.items()), columns=['gene', 'score'])
    ranking_df = ranking_df.sort_values(by='score', ascending=False)
    return ranking_df

def expand_gene_family(
    ranking_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Always divides the shap score evenly among family members.
    """
    expanded_rows = []
    for _, row in ranking_df.iterrows():
        gene = row['gene']
        score = row['score']
        if gene in gene_name_map:
            mapped_genes = gene_name_map[gene]
            score_per_gene = score / len(mapped_genes)
            for mgene in mapped_genes:
                expanded_rows.append({'gene': mgene.upper(), 'score': score_per_gene})
        else:
            expanded_rows.append({'gene': gene.upper(), 'score': score})

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df = expanded_df.groupby('gene', as_index=False).agg({'score': 'sum'})
    return expanded_df.sort_values(by='score', ascending=False)

def check_features_in_gene_sets(
    ranking_df_expanded: pd.DataFrame,
    gene_set_library: str
) -> Dict[str, Set[str]]:
    gene_sets = gp.get_library(name=gene_set_library, organism='Human')
    feature_presence = {}
    features_set = set(ranking_df_expanded['gene'].tolist())
    
    for gene in features_set:
        containing_sets = {gs for gs, genes in gene_sets.items() if gene in genes}
        feature_presence[gene] = containing_sets

    print(f"Genes NOT found in {gene_set_library}):")
    not_found_count = 0
    for _, row in ranking_df_expanded.iterrows():
        gene = row['gene']
        score = row['score']
        if not feature_presence.get(gene, set()):
            print(f"{gene:10s} | SHAP score: {score:.6f}")
            not_found_count += 1
    
    print(f"{not_found_count} were not found")

def perform_gsea_with_cleaned_genes(
    paper: str,
    base_dir_path: str = "./explainability/shapley/results",
    gene_set_library: str = "MSigDB_Hallmark_2020",
    save_dir: str = "./explainability/shapley/results/gsea_results",
    top_n: int = 100 #to specify the number of top features to use for GSEA in Transynergy
) -> pd.DataFrame:
    shap_values, inputs, feature_names, indices = load_shap_data(paper, base_dir_path)

    ranking_df = clean_and_aggregate_features(feature_names, shap_values, paper=paper)
    ranking_df_expanded = expand_gene_family(ranking_df)
    
    #check_features_in_gene_sets(ranking_df_expanded, gene_set_library)

    if paper == "transynergy":
        ranking_df_expanded = ranking_df_expanded.head(top_n)

    ranked_gene_list = ranking_df_expanded.set_index('gene')['score']

    # add tiny noise to break ties
    ranked_gene_list = ranked_gene_list + np.random.normal(0, 1e-8, size=len(ranked_gene_list))

    gsea_results = gp.prerank(
        rnk=ranked_gene_list,
        gene_sets=gene_set_library,
        outdir=None,
        permutation_num=1000,
        seed=42,
        verbose=False,
        min_size=1,
        max_size=1000
    )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{paper}_GSEA_{gene_set_library}.pkl")

    gsea_results.res2d.to_pickle(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GSEA on SHAP values for Biomining or TranSynergy.")
    parser.add_argument("--paper", choices=["biomining", "transynergy"], default="biomining",
                        help="Select paper type to analyze: 'biomining' or 'transynergy'")
    args = parser.parse_args()
    paper = args.paper

    for lib in [
        "MSigDB_Hallmark_2020",
        "Reactome_2022",
        "GO_Biological_Process_2021",
    ]:
        print(f"\nRunning GSEA with gene set library: {lib} for paper {paper}")
        
        perform_gsea_with_cleaned_genes(
            paper=paper,
            gene_set_library=lib,
        )
