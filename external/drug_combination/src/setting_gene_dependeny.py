import os
from time import time
import shutil

unit_test = False

working_dir = os.getcwd()
src_dir = os.path.join(working_dir, 'src')
data_src_dir = os.path.join(working_dir, 'data')
# propagation_methods: target_as_1, RWlike, random_walk
propagation_method = 'random_walk'
# feature type: LINCS1000, others, determine whether or not ignoring drugs without hidden representation
feature_type = 'more'
F_repr_feature_length = 1000

activation_method =["relu"]
dropout = [0.2, 0.1, 0.1]
start_lr = 0.0001
lr_decay = 0.00001
model_type = 'mlp'
FC_layout = [256] * 1 + [64] * 1
n_epochs = 700
batch_size = 128
loss = 'mse'
NBS_logfile = os.path.join(working_dir, 'NBS_logfile')
data_specific = '_2401_0.5_norm_drug_target_36_norm_gd_singlet_whole_network_no_mean_cl50_all_more_cl'
data_folder = os.path.join(working_dir, 'datas' + data_specific)
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    open(os.path.join(data_folder, "__init__.py"), 'w+').close()

uniq_part = "_run_{!r}".format(int(time()))
run_dir = os.path.join(working_dir, uniq_part)
logfile = os.path.join(run_dir, 'logfile')

run_specific_setting = os.path.join(run_dir, "setting.py")
cur_dir_setting = os.path.join(src_dir, "setting.py")

if not os.path.exists(run_dir):
    os.makedirs(run_dir)
    open(os.path.join(run_dir, "__init__.py"), 'w+').close()
    shutil.copyfile(cur_dir_setting, run_specific_setting)

update_final_index = True
final_index = os.path.join(data_src_dir,"synergy_score/final_index.csv")
update_xy = False
old_x = os.path.join(data_src_dir,"synergy_score/x.npy")
old_x_lengths = os.path.join(data_src_dir,"synergy_score/old_x_lengths.pkl")
old_y = os.path.join(data_src_dir,"synergy_score/y.pkl")

y_labels_file = os.path.join(src_dir, 'y_labels.p')
### ecfp, phy, ge, gd
catoutput_output_type = data_specific + "_dt"
save_final_pred = True
#["ecfp", "phy", "ge", "gd"]
catoutput_intput_type = [data_specific + "_dt"]
#{"ecfp": 2048, "phy": 960, "single": 15, "proteomics": 107}
dir_input_type = {}#{"single": 15, "proteomics": 107}

neural_fp = True
chemfp_drug_feature_file = os.path.join(data_src_dir, 'chemicals', 'drug_features_all_three_tanh.csv')
chem_linear_layers = [1024]
drug_input_dim = {'atom': 62, 'bond': 6}
conv_size = [16, 16]
degree = [0, 1, 2, 3, 4, 5]
drug_emb_dim = 512

genes = os.path.join(data_src_dir, 'Genes', 'genes_2401_df.csv')
synergy_score = os.path.join(data_src_dir, 'synergy_score', 'synergy_score.csv')
pathway_dataset = os.path.join(data_src_dir, 'pathways', 'genewise.p')
cl_genes_dp = os.path.join(data_src_dir, 'cl_gene_dp', 'new_gene_dependencies_35.csv')
#genes_network = '../genes_network/genes_network.csv'
#drugs_profile = '../drugs_profile/drugs_profile.csv'
L1000_upregulation = os.path.join(data_src_dir, 'F_repr', 'sel_F_drug_sample.csv')
L1000_downregulation = os.path.join(data_src_dir, 'F_repr', 'sel_F_drug_sample_1.csv')
add_single_response_to_drug_target = True
F_cl = os.path.join(data_src_dir, 'F_repr', 'sel_F_cl_sample.csv')
single_response = os.path.join(data_src_dir, 'chemicals', 'single_response_features.csv')

drug_ECFP = os.path.join(data_src_dir, 'chemicals', 'ECFP6.csv')
drug_physicochem = os.path.join(data_src_dir, 'chemicals', 'physicochemical_des.csv')
cl_ECFP = os.path.join(data_src_dir, 'RF_features', 'features_importance_df.csv')
cl_physicochem = os.path.join(data_src_dir, 'RF_features', 'features_importance_df_phychem.csv')
inchi_merck = os.path.join(data_src_dir, 'chemicals', 'inchi_merck.csv')

# networks: string_network, all_tissues_top
network_update = True
network_prop_normalized = True
network_path = os.path.join(data_src_dir, 'network')
network = os.path.join(data_src_dir, 'network', 'string_network')
network_matrix = os.path.join(data_src_dir, 'network', 'string_network_matrix.csv')
split_random_seed = 3
index_in_literature = True
index_renewal = True
train_index = os.path.join(data_src_dir, 'train_index_' + str(split_random_seed))
test_index = os.path.join(data_src_dir, 'test_index_' + str(split_random_seed))

renew = False
gene_expression_simulated_result_matrix = os.path.join(data_src_dir, 'chemicals', 'gene_expression_simulated_result_matrix_string.csv')
random_walk_simulated_result_matrix = os.path.join(data_src_dir, 'chemicals', 'random_walk_simulated_result_matrix_2401_0.5_norm_36_whole_network_no_mean')
intermediate_ge_target0_matrix = os.path.join(data_src_dir, 'chemicals', 'intermediate_ge_target0_matrix')

ml_train = False
test_ml_train = False

# estimators: RandomForest, GradientBoosting
estimator = "RandomForest"

if not os.path.exists(os.path.join(src_dir, 'tensorboard_logs')):
    os.mkdir(os.path.join(src_dir, 'tensorboard_logs'))
tensorboard_log = os.path.join(src_dir, "tensorboard_logs/{}".format(time()))

combine_gene_expression_renew = False
gene_expression = "Gene_expression_raw/normalized_gene_expession_35.tsv" #"CCLE.tsv"
backup_expression = "Gene_expression_raw/normalized_gene_expession_35.tsv" #"GDSC.tsv"
netexpress_df = "Gene_expression_raw/netexpress_scores_norm.tsv"

raw_expression_data_renew = False
processed_expression_raw = os.path.join(data_src_dir, 'Gene_expression_raw', 'processed_expression_raw')

combine_drug_target_renew = False
combine_drug_target_matrix = os.path.join(data_src_dir, 'chemicals', 'combine_drug_target_matrix.csv')

drug_profiles_renew = False
drug_profiles = os.path.join(data_src_dir, 'chemicals','new_dedup_drug_profile.csv')

python_interpreter_path = '/Users/QiaoLiu1/anaconda3/envs/pynbs_env/bin/python'

y_transform = True

### ['drug_target_profile', 'drug_ECFP', 'drug_physiochemistry', 'drug_F_repr']
drug_features = ['drug_target_profile']
#drug_features = ['drug_F_repr']
ecfp_phy_drug_filter_only = True
save_each_ecfp_phy_data_point = True

### ['gene_dependence', 'netexpress','gene_expression', 'cl_F_repr', 'cl_ECFP', 'cl_drug_physiochemistry', 'combine_drugs_for_cl']
cellline_features = ['gene_dependence']
#cellline_features = ['cl_F_repr' ]

one_linear_per_dim = True

single_response_feature = []#['single_response']

#arrangement = [[1,5,11],[2,6,12],[0,4,8],[0,4,9]]
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
old_model_path = os.path.join(working_dir, "_run_1582753440/best_model__2401_0.8_norm_drug_target_36_norm_net_single")

get_feature_imp = False
save_feature_imp_model = True
save_easy_input_only = (len(n_feature_type) == 1)
save_out_imp = False
save_inter_imp = False
best_model_path = os.path.join(run_dir, "best_model_" + data_specific)
input_importance_path = os.path.join(working_dir, "input_importance_" + data_specific)
out_input_importance_path = os.path.join(working_dir, "out_input_importance_" + data_specific)
transform_input_importance_path = os.path.join(working_dir, "transform_input_importance_" +data_specific)
feature_importance_path = os.path.join(working_dir, 'all_features_importance_' + data_specific )
