import os

cwd = os.getcwd()
prediction_training = os.path.join(cwd, "shap_analysis_dat/prediction_nest_label_2401_0.5_dedup_norm_drug_norm_net_dt_training")
prediction_testing = os.path.join(cwd, "shap_analysis_dat/prediction_nest_label_2401_0.5_dedup_norm_drug_norm_net_dt_testing")

importance_files = ["shap_analysis_dat/input_importance_nest_label_2401_0.5_dedup_norm_drug_norm_net"]
all_index_list_file = "shap_analysis_dat/all_index_list"

drug_target = os.path.join(cwd, "chemicals/new_raw_chemicals.csv")
gene_map_df_file = "Genes/genes_2401_df.csv"
gene_pickle_file = "shap_analysis_dat/gene2401_pkl_file.p"


saved_pdf_folder = "/Users/QiaoLiu1/drug_combine_graphs/net_50"
if not os.path.exists(saved_pdf_folder):
    os.mkdir(saved_pdf_folder)