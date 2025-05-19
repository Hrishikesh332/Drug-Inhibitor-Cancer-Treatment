import logging
import os
from dataclasses import dataclass
from time import time
from typing import Optional
from pathlib import Path

TRANS_SYNERGY_ROOT = Path("external/drug_combination")

_DATA_DIR = TRANS_SYNERGY_ROOT / "data"
_SRC_DIR = TRANS_SYNERGY_ROOT / "trans_synergy"
_RUNS_DIR = TRANS_SYNERGY_ROOT / "runs"

logger = logging.getLogger(__name__)
@dataclass
class Settings:
    """
    Settings singleton that carries all configuration.
    """
    # Configurable values
    synergy_score: str
    cellline_features: tuple # ['gene_dependence', 'netexpress','gene_expression', 'cl_F_repr', 'cl_ECFP', 'cl_drug_physiochemistry', 'combine_drugs_for_cl']
    gene_expression: str
    backup_expression: str
    netexpress_df: str
    n_epochs: int = 800

    unit_test = False

    propagation_method = 'random_walk'  # target_as_1, RWlike, random_walk
    # feature type: LINCS1000, others, determine whether or not ignoring drugs without hidden representation
    feature_type = 'more'
    F_repr_feature_length = 1000

    activation_method =['ReLU'] # ['LeakyReLU', 'Tanh']
    dropout = [0.2, 0.1, 0.1]
    start_lr = 0.0001
    lr_decay = 0.00001
    model_type = 'mlp'
    FC_layout = [256] * 1 + [64] * 1
    batch_size = 128
    loss = 'mse'

    NBS_logfile = TRANS_SYNERGY_ROOT / 'NBS_logfile'
    logfile =  TRANS_SYNERGY_ROOT / 'logfile'

    data_specific = '_2401_0.5_norm_drug_target_36_norm_gd_singlet_whole_network_no_mean_cl50_all_more_cl'

    uniq_part = "_run_{!r}".format(int(time()))
    run_dir = os.path.join(_RUNS_DIR, uniq_part)

    update_final_index = True
    final_index = os.path.join(_DATA_DIR, "synergy_score/final_index.csv")
    update_xy = True
    old_x = os.path.join(_DATA_DIR, "synergy_score/x.npy")
    old_x_lengths = os.path.join(_DATA_DIR, "synergy_score/old_x_lengths.pkl")
    old_y = os.path.join(_DATA_DIR, "synergy_score/y.pkl")

    y_labels_file = os.path.join(_DATA_DIR, 'y_labels', 'y_labels.p')
    ### ecfp, phy, ge, gd
    catoutput_output_type = data_specific + "_dt"
    save_final_pred = True
    #["ecfp", "phy", "ge", "gd"]
    catoutput_intput_type = [data_specific + "_dt"]
    #{"ecfp": 2048, "phy": 960, "single": 15, "proteomics": 107}
    dir_input_type = {} #{"single": 15, "proteomics": 107}

    genes = os.path.join(_DATA_DIR, 'genes', 'genes_2401_df.csv')
    pathway_dataset = os.path.join(_DATA_DIR, 'pathways', 'genewise.p')
    cl_genes_dp = os.path.join(_DATA_DIR, 'cl_gene_dp', 'new_gene_dependencies_35.csv')

    L1000_upregulation = os.path.join(_DATA_DIR, 'F_repr', 'sel_F_drug_sample.csv')
    L1000_downregulation = os.path.join(_DATA_DIR, 'F_repr', 'sel_F_drug_sample_1.csv')
    add_single_response_to_drug_target = True
    F_cl = os.path.join(_DATA_DIR, 'F_repr', 'sel_F_cl_sample.csv')
    single_response = os.path.join(_DATA_DIR, 'chemicals', 'single_response_features.csv')

    drug_ECFP = os.path.join(_DATA_DIR, 'chemicals', 'ECFP6.csv')
    drug_physicochem = os.path.join(_DATA_DIR, 'chemicals', 'physicochemical_des.csv')
    cl_ECFP = os.path.join(_DATA_DIR, 'RF_features', 'features_importance_df.csv')
    cl_physicochem = os.path.join(_DATA_DIR, 'RF_features', 'features_importance_df_phychem.csv')

    # networks: string_network, all_tissues_top
    network_update = True
    network_prop_normalized = True
    network_path = os.path.join(_DATA_DIR, 'network')
    network = os.path.join(_DATA_DIR, 'network', 'string_network')
    network_matrix = os.path.join(_DATA_DIR, 'network', 'string_network_matrix.csv')
    split_random_seed = 3
    index_in_literature = True

    renew = False
    gene_expression_simulated_result_matrix = os.path.join(_DATA_DIR, 'chemicals', 'gene_expression_simulated_result_matrix_string.csv')
    random_walk_simulated_result_matrix = os.path.join(_DATA_DIR, 'chemicals', 'random_walk_simulated_result_matrix_2401_0.5_norm_36_whole_network_no_mean')
    intermediate_ge_target0_matrix = os.path.join(_DATA_DIR, 'chemicals', 'intermediate_ge_target0_matrix')

    ml_train = False
    test_ml_train = False

    estimator = "RandomForest"  # GradientBoosting

    combine_gene_expression_renew = False

    raw_expression_data_renew = False
    processed_expression_raw = os.path.join(_DATA_DIR, 'gene_expression_raw', 'processed_expression_raw_norm')

    combine_drug_target_renew = False
    combine_drug_target_matrix = os.path.join(_DATA_DIR, 'chemicals', 'combine_drug_target_matrix.csv')

    drug_profiles_renew = False
    drug_profiles = os.path.join(_DATA_DIR, 'chemicals', 'new_dedup_drug_profile.csv')

    # TODO: refactor this whole thing!
    python_interpreter_path = '/Users/QiaoLiu1/anaconda3/envs/pynbs_env/bin/python'

    y_transform = True

    drug_features = ['drug_target_profile'] # ['drug_ECFP', 'drug_physiochemistry', 'drug_F_repr']
    ecfp_phy_drug_filter_only = True
    save_each_ecfp_phy_data_point = True

    one_linear_per_dim = True

    single_response_feature = []#['single_response']

    expression_dependencies_interaction = False
    arrangement = [[0,1,2]]
    update_features = True
    output_FF_layers = [2000, 1000, 1]
    n_feature_type = [3]
    single_repsonse_feature_length = 10 * 2
    if 'single_response' not in single_response_feature:
        single_repsonse_feature_length = 0
    d_model_i = 1
    d_model_j = 400
    d_model = d_model_i * d_model_j
    attention_heads = 1
    attention_dropout = 0.2
    n_layers = 1 # This has to be 1

    load_old_model = False
    old_model_path = "_run_1582753440/best_model__2401_0.8_norm_drug_target_36_norm_net_single"

    get_feature_imp = False
    save_feature_imp_model = True
    save_easy_input_only = (len(n_feature_type) == 1)
    save_out_imp = False
    save_inter_imp = False
    best_model_path = os.path.join(run_dir, "best_model_" + data_specific)
    input_importance_path = "input_importance_" + data_specific
    out_input_importance_path = "out_input_importance_" + data_specific
    transform_input_importance_path = "transform_input_importance_" +data_specific
    feature_importance_path = 'all_features_importance_' + data_specific

    inchi_merck = os.path.join(_DATA_DIR, 'chemicals', 'inchi_merck.csv')

    neural_fp = True
    chemfp_drug_feature_file = os.path.join(_DATA_DIR, 'chemicals', 'drug_features_all_three_tanh.csv')
    chem_linear_layers = [1024]
    drug_input_dim = {'atom': 62, 'bond': 6}
    conv_size = [16, 16]
    degree = [0, 1, 2, 3, 4, 5]
    drug_emb_dim = 512

    perform_importance_study = False

    genes_dp_indexes_path = os.path.join(_DATA_DIR, "cl_gene_dp", "all_dependencies_genes_map.csv")
    raw_drug_target_profile_path = os.path.join(_DATA_DIR, "chemicals", "raw_chemicals.csv")




gene_dependency_setting = Settings(
    synergy_score=os.path.join(_DATA_DIR, 'synergy_score', 'synergy_score.csv'),
    cellline_features=('gene_dependence',),
    n_epochs=800,
    gene_expression="gene_expression_raw/normalized_gene_expession_35.tsv",
    backup_expression = "gene_expression_raw/normalized_gene_expession_35.tsv",
    netexpress_df = "gene_expression_raw/netexpress_scores_norm.tsv",
)

gene_expression_setting = Settings(
    synergy_score=os.path.join(_DATA_DIR, 'synergy_score', 'combin_data_35.csv'),
    cellline_features=('gene_expression',),
    n_epochs=800,
    gene_expression="gene_expression_raw/normalized_gene_expession_35.tsv",
    backup_expression = "gene_expression_raw/normalized_gene_expession_35.tsv",
    netexpress_df = "gene_expression_raw/netexpress_scores_norm.tsv",
)

net_setting = Settings(
    synergy_score=os.path.join(_DATA_DIR, 'synergy_score', 'combin_data_35.csv'),
    cellline_features=('netexpress',),
    n_epochs=1000,
    gene_expression = "gene_expression_raw/CCLE.tsv",
    backup_expression = "gene_expression_raw/GDSC.tsv",
    netexpress_df = "gene_expression_raw/netexpress_35_.tsv",
)

_active_settings: Optional[Settings] = gene_dependency_setting  # will hold the settings choice. default is gene dependency

# ----- public API -------------------------------------------------------------
def configure(conf: Settings) -> None:
    """
    Call once, as early as possible (e.g. in `main()` or your CLI runner) to configure settings for the application
    """
    global _active_settings
    _active_settings = conf



def get() -> Settings:
    """Typed accessor for the currently active settings object."""
    if _active_settings is None:
        raise RuntimeError("Settings not configured yet. Please run settings.configure(). See main function for example")
    return _active_settings
