from torch import load
from src import shap_analysis_setting
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import os

class SHAP_ANALYSIS:

    drug_target = None
    def __init__(self, data, index_list, genes):
        assert data.shape[0] == len(index_list), "data and index are not matching"
        assert data.shape[1] == 3, "more than 3 input feature sets"
        self.drug_a, self.drug_b, self.cell_line = pd.DataFrame(data[:,0,:], index=index_list, columns=genes), \
                                                   pd.DataFrame( data[:,1,:], index= index_list, columns= genes), \
                                                   pd.DataFrame( data[:,2,:], index= index_list, columns= genes)
        if self.drug_target is None:
            self.drug_target = pd.read_csv(shap_analysis_setting.drug_target, index_col=0)

    def plot_gene_wise_shap(self, index_name):


        shap_series = []
        shap_series.append(self.drug_a.loc[index_name, :])
        shap_series.append(self.drug_b.loc[index_name, :])
        shap_series.append(self.cell_line.loc[index_name, :])
        self.__rank_and_plot_values(shap_series, 10, index_name)

    def __rank_and_plot_values(self, shap_series, n, index_name):

        plt.style.use('seaborn-poster')
        plt.rcParams['font.family'] = 'serif'
        rcParams['font.sans-serif'] = ['Palatino']
        rcParams['figure.max_open_warning'] = 30
        drug_a, drug_b, cell_line, _ = index_name.split("_")
        drug_a_target = self.drug_target.loc[drug_a, 'combin_gene'].split(",")
        drug_b_target = self.drug_target.loc[drug_b, 'combin_gene'].split(",")
        drug_a_high_shap_gene = shap_series[0].sort_values(ascending=False)[:n].index
        drug_b_high_shap_gene = shap_series[1].sort_values(ascending=False)[:n].index
        cell_line_high_shap_gene = shap_series[2].sort_values(ascending=False)[:n].index
        graph_index = drug_a_target + drug_b_target + list(drug_a_high_shap_gene) + \
                      list(drug_b_high_shap_gene) + list(cell_line_high_shap_gene)
        plt.figure(figsize=(16, 16))
        plt.subplot(311)
        sns.barplot(x=shap_series[0].index, y=shap_series[0], order=graph_index, color="orange")
        plt.xticks(rotation=45, fontsize=12)
        for label in plt.gca().get_xticklabels():
            if label.get_text() in (self.drug_target.loc[drug_a, 'combin_gene'].split(",")):
                if label.get_text() in (self.drug_target.loc[drug_b, 'combin_gene'].split(",")):
                    label.set_color('r')
                    label.set_fontsize(15)
                else:
                    label.set_color('g')
                    label.set_fontsize(15)
            if label.get_text() in (self.drug_target.loc[drug_b, 'combin_gene'].split(",")):
                label.set_color('b')
                label.set_fontsize(15)
        plt.ylabel("SHAP values", fontsize = 15)
        plt.title(index_name + ": " + drug_a, fontsize=20)
        plt.subplot(312)
        sns.barplot(x=shap_series[1].index, y=shap_series[1], order=graph_index, color="orange")
        plt.xticks(rotation=45, fontsize=12)
        for label in plt.gca().get_xticklabels():
            if label.get_text() in (self.drug_target.loc[drug_a, 'combin_gene'].split(",")):
                if label.get_text() in (self.drug_target.loc[drug_b, 'combin_gene'].split(",")):
                    label.set_color('r')
                    label.set_fontsize(15)
                else:
                    label.set_color('g')
                    label.set_fontsize(15)
            if label.get_text() in (self.drug_target.loc[drug_b, 'combin_gene'].split(",")):
                label.set_color('b')
                label.set_fontsize(15)
        plt.ylabel("SHAP values", fontsize=20)
        plt.title(index_name + ": " + drug_b, fontsize=20)
        plt.subplot(313)
        sns.barplot(x=shap_series[2].index, y=shap_series[2], order=graph_index, color="orange")
        plt.xticks(rotation=45, fontsize=12)
        for label in plt.gca().get_xticklabels():
            if label.get_text() in (self.drug_target.loc[drug_a, 'combin_gene'].split(",")):
                if label.get_text() in (self.drug_target.loc[drug_b, 'combin_gene'].split(",")):
                    label.set_color('r')
                    label.set_fontsize(15)
                else:
                    label.set_color('g')
                    label.set_fontsize(15)
            if label.get_text() in (self.drug_target.loc[drug_b, 'combin_gene'].split(",")):
                label.set_color('b')
                label.set_fontsize(15)
        plt.ylabel("SHAP values", fontsize=20)
        plt.title(index_name + ": " + cell_line, fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_analysis_setting.saved_pdf_folder, "result_graph_" + index_name + ".pdf"))

def construct_map(df):

    map_dic = {}
    for row in df.iterrows():
        map_dic[row[1]['entrez']] = row[1]['symbol']
    return map_dic

if __name__ == "__main__":

    training_data = load(shap_analysis_setting.prediction_training)
    training_df = pd.DataFrame(training_data, columns=['combination', 'prediction', 'ground_truth'])
    testing_data = load(shap_analysis_setting.prediction_testing)
    testing_df = pd.DataFrame(testing_data[:len(testing_data)//2, :], columns=['combination', 'prediction', 'ground_truth'])

    alpha = 2
    testing_df['order'] = pd.to_numeric(testing_df['ground_truth']) - \
                          alpha * np.abs(pd.to_numeric(testing_df['prediction']) - pd.to_numeric(testing_df['ground_truth']))
    testing_df.sort_values(by='order', inplace=True, ascending=False)

    genes = list(pickle.load(open(shap_analysis_setting.gene_pickle_file, 'rb')))
    gene_map_df = pd.read_csv(shap_analysis_setting.gene_map_df_file)
    gene_map = construct_map(gene_map_df)
    genes_symbols = [gene_map[x] for x in genes]

    importance_data = pickle.load(open(shap_analysis_setting.importance_files[0], 'rb'))
    index_list = pickle.load(open(shap_analysis_setting.all_index_list_file, 'rb'))

    shap_analysis_cls = SHAP_ANALYSIS(importance_data, index_list, genes_symbols)
    count = 0
    for drug_combine in testing_df['combination']:
        if count>50:
            break
        if drug_combine in index_list:
            shap_analysis_cls.plot_gene_wise_shap(drug_combine)
            count+=1









