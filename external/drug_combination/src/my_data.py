import pandas as pd
from os import path, mkdir
import numpy as np
import torch
from torch.utils import data
from src import drug_drug, setting, network_propagation, logger
from sklearn.preprocessing import StandardScaler
from torch import save


class CustomDataLoader:
    pass

class CustomDataReader:
    pass


class GenesDataReader(CustomDataReader):

    genes = None
    def __init__(self):
        super().__init__()

    @classmethod
    def __genes_initializer(cls):

        if cls.genes is None:
            cls.genes = pd.read_csv(setting.genes,
                                    dtype={'entrez': np.int})
        assert {'symbol','entrez'}.issubset(set(cls.genes.columns)), \
            "Genes data frame columns name should have symbol and entrez"

    @classmethod
    def get_genes(cls):
        cls.__genes_initializer()
        return cls.genes

    @classmethod
    def get_gene_entrez_set(cls):

        ### return genes entrez ID set as integer type
        cls.__genes_initializer()
        return set(cls.genes['entrez'])

    @classmethod
    def get_gene_symbol_set(cls):
        cls.__genes_initializer()
        return set(cls.genes['symbol'])

class NetworkDataReader(CustomDataReader):

    raw_network=None
    network = None
    entrez_set = None

    def __init__(self):
        super().__init__()

    @classmethod
    def __raw_network_initializer(cls):
        if cls.raw_network is None:
            cls.raw_network = pd.read_csv(setting.network, header=None, sep ='\t')
            assert len(cls.raw_network.columns) == 3, "genes network file should have three columns"
            cls.raw_network.columns = ['entrez_a', 'entrez_b', 'association']
            cls.raw_network = cls.raw_network.astype({'entrez_a': np.int,
                                                      'entrez_b': np.int,
                                                      'association': np.float})

    @classmethod
    def __filter_network(cls):

        if cls.entrez_set is None:
            cls.entrez_set = GenesDataReader.get_gene_entrez_set()

        if cls.network is None:
            filter = (cls.raw_network['entrez_a'].isin(list(cls.entrez_set))) &\
                     (cls.raw_network['entrez_b'].isin(list(cls.entrez_set)))
            cls.network = cls.raw_network[filter]

    @classmethod
    def get_network(cls):
        cls.__raw_network_initializer()
        cls.__filter_network()
        return cls.network

    @classmethod
    def check_genes_in_network(cls):

        if cls.network is None:
            cls.get_network()
        ### Make sure genes are all in network
        unfound_genes = set(cls.entrez_set) - (set(cls.network['entrez_a']).union(set(cls.network['entrez_b'])))
        if len(unfound_genes) == 0:
            logger.info("Found all genes in networks")
        else:
            logger.info("Unfound genes in networks: %s" % str(unfound_genes))

class DrugTargetProfileDataLoader(CustomDataLoader):

    raw_drug_target_profile = None
    entrez_set = GenesDataReader.get_gene_entrez_set()
    network = None
    drug_target = None
    raw_simulated_drug_target = None
    simulated_drug_target_profile = None

    def __init__(self):
        super().__init__()

    @classmethod
    def __raw_drug_target_initializer(cls):
        if cls.raw_drug_target_profile is None:
            cls.raw_drug_target_profile = pd.read_csv(setting.working_dir + "/chemicals/raw_chemicals.csv")
            assert {'Name', 'combin_entrez'}.issubset(set(cls.raw_drug_target_profile.columns)), \
                "Name and combin_entrez should be in raw_drug_target_profile columns names"

    @classmethod
    def __create_drug_target_profiles(cls):

        # return data frame
        # columns: drugs, index: entrez_ID (int)
        cls.__raw_drug_target_initializer()
        drug_profile = pd.DataFrame(np.zeros(shape=(len(cls.entrez_set), len(cls.raw_drug_target_profile))),
                                    index=list(cls.entrez_set),
                                    columns=cls.raw_drug_target_profile['Name'])

        logger.debug("Creating raw drug target data frame")
        for row in cls.raw_drug_target_profile.iterrows():
            if not isinstance(row[1]['combin_entrez'], str):
                continue
            chem_name, target_list = row[1]['Name'], row[1]['combin_entrez'].split(",")
            for target in target_list:
                target = int(target)
                if target in cls.entrez_set:
                    drug_profile.loc[target, chem_name] = 1

        logger.debug("Create raw drug target data frame successfully")
        return drug_profile

    @classmethod
    def __get_drug_target_profiles(cls):

        ### return data frame
        ###         5-FU  ABT-888  AZD1775  BEZ-235  BORTEZOMIB  CARBOPLATIN
        ### 1400      1        0        0        0           0            0
        ### 19122     0        0        0        0           0            0
        ### 123       0        1        1        0           1            0
        ### 24533     1        1        1        1           0            1
        ### 222       0        0        0        0           1            0
        if cls.drug_target is None:
            if not setting.drug_profiles_renew and path.exists(setting.drug_profiles):
                cls.drug_target = pd.read_csv(setting.drug_profiles, index_col=0)
                cls.drug_target.index = cls.drug_target.index.astype(int)
                #assert set(cls.drug_target.index).issubset(cls.entrez_set), "Drug Profile index is not correct"

            else:
                cls.drug_target = cls.__create_drug_target_profiles()
                cls.drug_target.to_csv(setting.drug_profiles)
        return cls.drug_target

    @classmethod
    def check_unfound_genes_in_drug_target(cls):

        if cls.drug_target is None:
            cls.__get_drug_target_profiles()
        ### Make sure that drug target genes and gene dependencies genes are in selected ~2300 genes
        logger.info("merged_drug_targets: %s" % str(cls.drug_target.head()))
        unfound_genes = set(cls.drug_target.index) - set(cls.entrez_set)
        if len(unfound_genes) == 0:
            logger.info("Found all genes in drug target")
        else:
            logger.info("Unfound genes in drug target: %s" % str(unfound_genes))

    @classmethod
    def __get_simulated_drug_target_profiles(cls):

        ### return dataframe:
        ### columns=genes['entrez'], index=drugs
        if cls.network is None:
            cls.network = NetworkDataReader.get_network()
        if cls.drug_target is None:
            cls.__get_drug_target_profiles()

        logger.debug("Network propagation (methods: {}) ... ".format(setting.propagation_method))
        simulated_drug_target_matrix = network_propagation.random_walk_network_propagation(
                setting.random_walk_simulated_result_matrix)

        logger.debug("Network propagation (methods: {}) is Done.".format(setting.propagation_method))
        assert set(simulated_drug_target_matrix.columns).issubset(cls.entrez_set), \
            'simulated drug target profile data frame columns names are not correct (types or contents)'
        return simulated_drug_target_matrix

    @classmethod
    def get_filtered_simulated_drug_target_matrix(cls):

        ### return dataframe:
        ### columns=genes['entrez'], index=drugs
        if cls.simulated_drug_target_profile is None:
            if cls.raw_simulated_drug_target is None:
                cls.raw_simulated_drug_target = cls.__get_simulated_drug_target_profiles()
            ###indexwises filter
            index_filter = ~(cls.raw_simulated_drug_target == 0).all(axis = 1)
            col_filter = (cls.raw_simulated_drug_target.var(axis=0) > 0)
            logger.debug("Removed {!r} drugs".format(sum(~index_filter)))
            logger.debug("Removed {!r} genes".format(sum(~col_filter)))
            cls.simulated_drug_target_profile = cls.raw_simulated_drug_target.loc[index_filter, col_filter]
        return cls.simulated_drug_target_profile

    #
    # gene_filter = (simulated_drug_target.var(axis=0) > 0)
    # sel_drug_target = simulated_drug_target.loc[:, gene_filter]
    @classmethod
    def get_sel_drugs_set(cls):

        ### return set: set of selected drugs name
        drug_target_matrix = cls.get_filtered_simulated_drug_target_matrix()
        return set(drug_target_matrix.index)

    @classmethod
    def check_drugs_in_drug_target(cls):

        exp_drugs = cls.get_sel_drugs_set()
        if cls.drug_target is None:
            cls.__get_drug_target_profiles()
        ### Make sure drugs are all in drug_target dataframe
        unfound_drugs = exp_drugs - set(cls.drug_target.columns)
        if len(unfound_drugs) == 0:
            logger.info("Found all Drugs")
        else:
            logger.info("Unfound Drugs: %s" % str(unfound_drugs))

class SynergyDataReader(CustomDataReader):

    synergy_score = None
    sel_drugs = None
    drugs_filtered = False
    final_index = None

    def __init__(self):
        super().__init__()

    ### Reading synergy score data and return data frame ###
    ### Unnamed: 0,drug_a_name,drug_b_name,cell_line,synergy
    ### 5-FU_ABT-888_A2058,5-FU,ABT-888,A2058,7.6935301658
    ### 5-FU_ABT-888_A2780,5-FU,ABT-888,A2780,7.7780530601
    @classmethod
    def __initialize_synergy_score(cls):
        if cls.synergy_score is None:
            cls.synergy_score = pd.read_csv(setting.synergy_score)
            assert {'cell_line', 'drug_a_name', 'drug_b_name'}.issubset(set(cls.synergy_score.columns)), \
                "'cell_line', 'drug_a_name', 'drug_b_name' are not in synergy score data frame"

    @classmethod
    def __filter_drugs(cls):
    ### Some drugs are removed because the drug feature vectors only have zero
        if cls.drugs_filtered:
            return
        if setting.feature_type == 'LINCS1000':
            cls.sel_drugs = set(list(pd.read_csv(setting.L1000_upregulation, header = None, index_col=0).index))
        elif setting.feature_type == 'others':
            cls.sel_drugs = DrugTargetProfileDataLoader.get_sel_drugs_set() & set(list(pd.read_csv(
                setting.L1000_upregulation, header = None, index_col=0).index))
        else:
            cls.sel_drugs = DrugTargetProfileDataLoader.get_sel_drugs_set()

        filter1 = (cls.synergy_score['drug_a_name'].isin(cls.sel_drugs)) & (cls.synergy_score['drug_b_name'].isin(cls.sel_drugs))
        cls.synergy_score = cls.synergy_score[filter1]
        logger.debug("Post filteration, synergy score has {!r} data points".format(len(cls.synergy_score)))
        cls.drugs_filtered = True

    @classmethod
    def get_synergy_score(cls, pro_filter = False):

        if cls.synergy_score is None:
            cls.__initialize_synergy_score()
        filtered = cls.drugs_filtered
        if not filtered:
            cls.__filter_drugs()
        cls.synergy_score = cls.synergy_score.reset_index(drop=True)
        if pro_filter:
            proteomics_filter = ~cls.synergy_score['cell_line'].isin({'DLD1', 'OCUBM', 'A427'})
            cls.synergy_score = cls.synergy_score[proteomics_filter]
        #cls.synergy_score.to_csv("./synergy_score/filtered_data.csv", index = False)
        return cls.synergy_score

    @classmethod
    def get_synergy_data_cell_lines(cls):
        filtered = cls.drugs_filtered
        if not filtered:
            cls.__filter_drugs()
        return set(cls.synergy_score['cell_line'])

    @classmethod
    def get_synergy_data_drugs(cls):
        filtered = cls.drugs_filtered
        if not filtered:
            cls.__filter_drugs()
        return set(cls.synergy_score['drug_a_name']).union(set(cls.synergy_score['drug_b_name']))

    @classmethod
    def get_final_index(cls, pro_filter = False):

        if cls.final_index is None:
            synergy_score = cls.get_synergy_score(pro_filter = pro_filter)
            final_index_1 = synergy_score.reset_index().apply(
                lambda row: row['drug_a_name'] + '_' + row['drug_b_name'] + '_' +
                            row['cell_line'] + '_' + str(row['index']), axis=1)
            final_index_2 = synergy_score.reset_index().apply(
                lambda row: row['drug_b_name'] + '_' + row['drug_a_name'] + '_' +
                            row['cell_line'] + '_' + str(row['index']), axis=1)
            cls.final_index = pd.concat([final_index_1, final_index_2], axis=0).reset_index(drop=True)
        return cls.final_index

class GeneDependenciesDataReader(CustomDataReader):

    genes_dp_indexes = None
    genes_dp = None
    exp_cell_lines = None
    exp_genes = None
    cell_line_filtered = False
    gene_filtered = False
    var_filtered = False

    def __init__(self):
        super().__init__()

    @classmethod
    def __initialize_genes_dp_indexes(cls):
        if cls.genes_dp_indexes is None:
            cls.genes_dp_indexes = pd.read_csv(setting.working_dir + "/cl_gene_dp/all_dependencies_genes_map.csv",
                                               usecols=['symbol', 'entrez'], dtype={'entrez': np.int})
    @classmethod
    def __initialize_genes_dp(cls):

        cls.__initialize_genes_dp_indexes()
        if cls.genes_dp is None:
            cls.genes_dp = pd.read_csv(setting.cl_genes_dp)
            if cls.genes_dp.shape[0] < 40:
                cls.genes_dp = cls.genes_dp.set_index('Unnamed: 0').T
                cls.genes_dp.index = [int(x.split("(")[1][:-1]) for x in list(cls.genes_dp.index)]
            else:
                cls.genes_dp.index = cls.genes_dp_indexes['entrez']
                cls.genes_dp.columns = list(map(lambda x: x.split("_")[0], cls.genes_dp.columns))

    @classmethod
    def __filter_cell_line(cls):

        ### select only the cell lines studied
        if cls.cell_line_filtered:
            return
        if cls.exp_cell_lines is None:
            cls.exp_cell_lines = SynergyDataReader.get_synergy_data_cell_lines()
        col_filter = list(cls.exp_cell_lines)
        cls.genes_dp = cls.genes_dp[col_filter]
        cls.cell_line_filtered = True

    @classmethod
    def __filter_gene(cls):

        ### select only the genes focused on
        if cls.gene_filtered:
            return
        if cls.exp_genes is None:
            cls.exp_genes = GenesDataReader.get_gene_entrez_set()
        index_filter_1 = list(set(cls.exp_genes))
        cls.genes_dp = cls.genes_dp.groupby(level = 0).mean()
        cls.genes_dp = cls.genes_dp.loc[cls.genes_dp.index.intersection(index_filter_1)].reindex(index_filter_1)
        cls.gene_filtered = True
        print(len(index_filter_1))
        assert len(index_filter_1) != 0 and isinstance(index_filter_1[0], np.int), "entrezID filter should be integer"

    @classmethod
    def __rm_duplications(cls):

        ### average all the gene dependencies value belong to one entrezID
        ### cls.genes_dp = cls.genes_dp[list(cell_lines)].reset_index().
        ### drop_duplicates(subset='entrez').set_index('entrez')
        cls.genes_dp = cls.genes_dp.groupby(level=0).mean()

    @classmethod
    def __filter_var(cls):

        ### filter the data with variance at 0
        if cls.var_filtered:
            return

        sel_dp_filter = (cls.genes_dp.var(axis=1) > 0)
        cls.genes_dp = cls.genes_dp.loc[sel_dp_filter, :]
        cls.var_filtered = True

    @classmethod
    def get_gene_dp(cls):

        if cls.genes_dp is None:
            cls.__initialize_genes_dp()

        cls.__filter_cell_line()
        cls.__filter_gene()
        cls.__filter_var()
        cls.__rm_duplications()
        return cls.genes_dp

    @classmethod
    def check_unfound_genes_in_gene_dp(cls):

        if cls.genes_dp is None:
            cls.__initialize_genes_dp()
            cls.__filter_cell_line()
            cls.__filter_gene()
            cls.__filter_var()
            cls.__rm_duplications()

        logger.info("sel_dp: %s" % str(cls.genes_dp.head()))

        unfound_genes = set(cls.genes_dp.index) - set(GenesDataReader.get_gene_entrez_set())
        if len(unfound_genes) == 0:
            logger.info("Found all genes in genes dependencies")
        else:
            logger.info("Unfound genes in genes dependencies: %s" % str(unfound_genes))

    @classmethod
    def check_celllines_in_gene_dp(cls):

        ### Make sure cell lines in synergy score dataframe are in that in dependencies scores
        cell_lines = SynergyDataReader.get_synergy_data_cell_lines()
        unfound_cl = set(cls.genes_dp.columns) - cell_lines - {'symbol', 'entrez'}
        if len(unfound_cl) == 0:
            logger.info("Found all cell lines")
        else:
            logger.info("Unfound cell lines: %s" % str(unfound_cl))

class ExpressionDataLoader(CustomDataLoader):

    gene_expression = None
    backup_expression = None

    def __init__(self):
        super().__init__()

    @classmethod
    def __initialize_gene_expression(cls):
        ### make sure only one gene expression data frame is instantiated in this class
        ### return: gene expression data frame

        if cls.gene_expression is None:
            cls.gene_expression = pd.read_csv(setting.gene_expression, sep='\t')
            logger.debug("Read in gene expresion data successfully")
            cls.gene_expression.set_index(keys='Entrez', inplace=True)
        return cls.gene_expression

    @classmethod
    def __initialize_backup_expression(cls):
        ### make sure only one backup expression data frame is instantiated in this class
        ### data will firstly searched in CCLE database and then in GDSC database
        ### return: gene expression data frame

        if cls.backup_expression is None:
            cls.backup_expression = pd.read_csv(setting.backup_expression, sep='\t')
            logger.debug("Read in back up expresion data successfully")
            cls.backup_expression.set_index(keys='Entrez', inplace=True)
        return cls.backup_expression

    @classmethod
    def __filter_genes(cls, df, entrezIDs):

        ### genes: interested genes
        ### return data frame: Select only the genes interested in the data frame

        result_df = df.loc[df.index.intersection(entrezIDs)].reindex(entrezIDs)
        repo_genes, interested_genes = set(df.index), set(entrezIDs)
        if not repo_genes.issuperset(interested_genes):
            unfound = interested_genes - repo_genes
            logger.debug("{!r} are not found!".format(unfound))

        result_df.fillna(0, inplace=True)
        return result_df

    @classmethod
    def __filter_celllines(cls, df, celllines):

        ### cell line: interested cell lines
        ### return data frame: select only the cell lines interested by user
        result_df = df.loc[:, list(set(celllines) & set(df.columns))]
        repo_celllines, interested_celllines, unfound = set(df.columns), set(celllines), {}

        if not repo_celllines.issuperset(interested_celllines):
            unfound = interested_celllines - repo_celllines
            logger.debug("{!r} are not found!".format(unfound))

        if len(unfound):
            ### use the back up expression dataframe data
            cls.__initialize_backup_expression()
            backup_celllines_repo = set(cls.backup_expression.columns)
            if len(unfound.intersection(backup_celllines_repo)):
                more_cellline_df = cls.__filter_celllines(cls.backup_expression, list(unfound))
                result_df = pd.concat([result_df, more_cellline_df], axis=1)

        result_df.fillna(0, inplace=True)
        return result_df
    @classmethod
    def prepare_expresstion_df(cls, entrezIDs, celllines):

        ### entrezIDs, celllines: selection criterials
        ### return data frame: data frame that have interested cell lines and genes
        ###              A375   ..... (celllines)
        ###   1003(entrez)
        ###    ...

        cls.__initialize_gene_expression()

        result_df = cls.__filter_celllines(cls.gene_expression, celllines)
        result_df = cls.__filter_genes(result_df, entrezIDs)
        if setting.raw_expression_data_renew or not path.exists(setting.processed_expression_raw):
            logger.debug("Persist gene expression data frame")
            result_df.to_csv(setting.processed_expression_raw, index = False)

        return result_df

class NetExpressDataLoader(CustomDataLoader):

    netexpress_df = None
    def __init__(self):
        super().__init__()

    @classmethod
    def __initialize_gene_expression(cls):
        ### make sure only one gene expression data frame is instantiated in this class
        ### return: gene expression data frame

        if cls.netexpress_df is None:
            cls.netexpress_df = pd.read_csv(setting.netexpress_df, sep='\t')
            logger.debug("Read in netexpress data successfully")
        return cls.netexpress_df

    @classmethod
    def __filter_genes(cls, df, entrezIDs):

        ### genes: interested genes
        ### return data frame: Select only the genes interested in the data frame

        result_df = df.loc[df.index.intersection(entrezIDs)].reindex(entrezIDs)
        repo_genes, interested_genes = set(df.index), set(entrezIDs)
        if not repo_genes.issuperset(interested_genes):
            unfound = interested_genes - repo_genes
            logger.debug("{!r} are not found!".format(unfound))

        result_df.fillna(0, inplace=True)
        return result_df

    @classmethod
    def prepare_netexpress_df(cls, entrezIDs):

        ### entrezIDs, celllines: selection criterials
        ### return data frame: data frame that have interested cell lines and genes
        ###              A375   ..... (celllines)
        ###   1003(entrez)
        ###    ...

        cls.__initialize_gene_expression()
        result_df = cls.__filter_genes(cls.netexpress_df, entrezIDs)

        return result_df


class ECFPDataLoader(CustomDataLoader):

    drug_ECFP = None
    cl_ECFP = None

    def __init__(self):
        super().__init__()

    @classmethod
    def __dataloader_initializer(cls):

        if cls.drug_ECFP is None:
            cls.drug_ECFP = pd.read_csv(setting.drug_ECFP)
        if cls.cl_ECFP is None:
            cls.cl_ECFP = pd.read_csv(setting.cl_ECFP, index_col=0)

    @classmethod
    def get_drug_ecfp_data(cls, save_each_data_point = setting.save_each_ecfp_phy_data_point):

        if cls.drug_ECFP is None:
            cls.__dataloader_initializer()
        cls.drug_ECFP = cls.drug_ECFP[['Name', 'ECFP_6']]
        cls.drug_ECFP.set_index('Name', inplace= True)
        cls.drug_ECFP = cls.drug_ECFP['ECFP_6'].apply(lambda i: pd.Series(list(i))).astype(int)
        cls.drug_ECFP.columns = cls.drug_ECFP.columns.astype(str)
        #cls.ECFP = cls.ECFP.loc[:,~((cls.drug_ECFP==0).all(axis = 0))]
        cls.drug_ECFP = cls.drug_ECFP.loc[:, cls.__get_ecfp_filter(drug_filter_only=setting.ecfp_phy_drug_filter_only)]
        if save_each_data_point:
            if not path.exists("ecfp_datas"):
                mkdir("ecfp_datas")
            for i, one_drug_ecfp in enumerate(cls.drug_ECFP.values):
                save(one_drug_ecfp, path.join("ecfp_datas", cls.drug_ECFP.index[i] + '.pt'))
        return cls.drug_ECFP

    @classmethod
    def get_cl_ecfp_data(cls):

        if cls.cl_ECFP is None:
            cls.__dataloader_initializer()
        cls.cl_ECFP = cls.cl_ECFP.loc[:, cls.__get_ecfp_filter()]
        return cls.cl_ECFP

    @classmethod
    def __get_ecfp_filter(cls, drug_filter_only = False):

        if cls.cl_ECFP is None or cls.drug_ECFP is None:
            cls.__dataloader_initializer()
        drug_filter = (~((cls.drug_ECFP==0).all(axis = 0)))
        cl_filter = (~((cls.cl_ECFP==0).all(axis = 0)))
        common_filter = (drug_filter & cl_filter)
        if drug_filter_only:
            return drug_filter
        return common_filter

class PhysicochemDataLoader(CustomDataLoader):

    drug_physicochem = None
    cl_physicochem = None

    def __init__(self):
        super().__init__()

    @classmethod
    def __dataloader_initializer(cls):

        if cls.drug_physicochem is None:
            cls.drug_physicochem = pd.read_csv(setting.drug_physicochem, index_col=0)
        if cls.cl_physicochem is None:
            cls.cl_physicochem = pd.read_csv(setting.cl_physicochem, index_col = 0)

    @classmethod
    def get_drug_physicochem_property(cls, save_each_data_point = setting.save_each_ecfp_phy_data_point):

        if cls.drug_physicochem is None:
            cls.__dataloader_initializer()
        cls.drug_physicochem.drop('SMILE', inplace=True, axis=1)
        #cls.physicochem = cls.physicochem.loc[:, ~((cls.physicochem == 0).all(axis=0))]
        cls.drug_physicochem = cls.drug_physicochem.loc[:, cls.__get_physicochem_filter(drug_filter_only=setting.ecfp_phy_drug_filter_only)]
        physicochem_scaler = StandardScaler(with_mean=False)
        physicochem = physicochem_scaler.fit_transform(cls.drug_physicochem)
        physicochem = pd.DataFrame(physicochem, index=cls.drug_physicochem.index, columns=cls.drug_physicochem.columns)
        cls.drug_physicochem = physicochem
        if save_each_data_point:
            if not path.exists("phy_datas"):
                mkdir("phy_datas")
            for i, one_drug_phy in enumerate(cls.drug_physicochem.values):
                save(one_drug_phy, path.join("phy_datas", cls.drug_physicochem.index[i] + '.pt'))
        return cls.drug_physicochem

    @classmethod
    def get_cl_physiochem_property(cls):

        if cls.cl_physicochem is None:
            cls.__dataloader_initializer()
        cls.cl_physicochem = cls.cl_physicochem.loc[:, cls.__get_physicochem_filter()]
        return cls.cl_physicochem

    @classmethod
    def __get_physicochem_filter(cls, drug_filter_only = False):

        if cls.drug_physicochem is None or cls.cl_physicochem is None:
            cls.__dataloader_initializer()
        drug_filter = ~((cls.drug_physicochem == 0).all(axis=0))
        cl_filter = ~((cls.cl_physicochem == 0).all(axis=0))
        common_filter = drug_filter | cl_filter
        if drug_filter_only:
            return drug_filter
        return common_filter

class SingleResponseDataLoader(CustomDataLoader):

    single_response = None

    def __init__(self):
        super().__init__()

    @classmethod
    def __dataloader_initializer(cls):

        if cls.single_response is None:
            cls.single_response = pd.read_csv(setting.single_response, index_col=0).drop(['mean', 'sigma'], axis=1)
            cls.single_response['drug'] = cls.single_response['drug'].str.upper()
            cls.single_response.set_index(['cell_line', 'drug'], inplace = True)

    @classmethod
    def get_single_response(cls, save_each_data_point = setting.save_each_ecfp_phy_data_point):

        if cls.single_response is None:
            cls.__dataloader_initializer()
        if save_each_data_point:
            if not path.exists("single_datas"):
                mkdir("single_datas")
            for i, one_drug_single in enumerate(cls.single_response.values):
                save(one_drug_single, path.join("single_datas", "_".join(cls.single_response.index[i]) + '.pt'))
        return cls.single_response

class ProteomicsDataLoader(CustomDataLoader):

    proteomics = None

    def __init__(self):
        super().__init__()

    @classmethod
    def __dataloader_initializer(cls):

        if cls.proteomics is None:
            cls.proteomics = pd.read_csv(setting.ccle_pro, index_col=0)

    @classmethod
    def get_proteomics(cls, save_each_data_point = setting.save_each_ecfp_phy_data_point):

        if cls.proteomics is None:
            cls.__dataloader_initializer()
        if save_each_data_point:
            if not path.exists("proteomics_datas"):
                mkdir("proteomics_datas")
            for i, one_cl_pro in enumerate(cls.proteomics.values):
                save(one_cl_pro, path.join("proteomics_datas", cls.proteomics.index[i]) + '.pt')
        return cls.proteomics

class RepresentationSamplesDataLoader(CustomDataLoader):

    L1000_upregulation = None
    F_cl = None
    synergy_score = None
    data_initialized = False
    drug_a_features = None
    drug_b_features = None
    cellline_features = None
    whole_df = None

    def __init__(self):
        super.__init__()

    @classmethod
    def __dataloader_initializer(cls):

        if cls.data_initialized:
            return

        ######################
        ### 5-FU ....
        #####################
        cls.L1000_downregulation = pd.read_csv(setting.L1000_upregulation, header = None, index_col = 0)

        ######################
        ### A2058 ......
        #####################
        cls.F_cl = pd.read_csv(setting.F_cl, header = None, index_col = 0)

        ### Reading synergy score data ###
        ### Unnamed: 0,drug_a_name,drug_b_name,cell_line,synergy
        ### 5-FU_ABT-888_A2058,5-FU,ABT-888,A2058,7.6935301658
        ### 5-FU_ABT-888_A2780,5-FU,ABT-888,A2780,7.7780530601
        cls.synergy_score = SynergyDataReader.get_synergy_score()
        cls.data_initialized = True

    @classmethod
    def __features_prep(cls):

        ### generate drugs features
        if cls.drug_a_features is None or cls.drug_b_features is None or cls.cellline_features is None:
            cls.__dataloader_initializer()
            cls.drug_a_features = cls.L1000_upregulation.loc[list(cls.synergy_score['drug_a_name']), :].reset_index(drop=True)
            #cls.drug_a_features.fillna(0, inplace=True)
            cls.drug_b_features = cls.L1000_upregulation.loc[list(cls.synergy_score['drug_b_name']), :].reset_index(drop=True)
            #cls.drug_b_features.fillna(0, inplace=True)
            cls.cellline_features = cls.F_cl.loc[list(cls.synergy_score['cell_line']), :].reset_index(drop=True)
            #cls.cellline_features.fillna(0, inplace=True)
        return [cls.drug_a_features, cls.drug_b_features, cls.cellline_features]

    @classmethod
    def __construct_whole_raw_X(cls):

        ### return dataframe
        ###  first_half_drugs_features                first_half_cellline_features
        ###  switched_second_half_drugs_features      second_half_cellline_features
        if cls.whole_df is None:
            features_list = cls.__features_prep()
            first_half = pd.concat(features_list, axis=1)
            second_half = pd.concat([features_list[1], features_list[0], features_list[2]], axis=1)
            cls.whole_df = pd.concat([first_half, second_half], axis=0).reset_index(drop=True)
        return cls.whole_df

    @classmethod
    def Raw_X_features_prep(cls, methods):

        ### Generate final raw features dataset
        ### return: ndarray (n_samples, n_type_features, feature_dim) if 'attn'
        ###         ndarray (n_samples, n_type_features * feature_dim) else
        raw_x = cls.__construct_whole_raw_X().values
        if methods == 'attn':
            x = raw_x.reshape(-1, 3, setting.F_repr_feature_length)

        else:
            drug_features_len = int(1 / setting.n_feature_type * raw_x.shape[1])
            cl_features_len = int(raw_x.shape[1] - 2 * drug_features_len)
            assert cl_features_len == int((1 - 2 / setting.n_feature_type) * raw_x.shape[1]), \
                "features len are calculated in wrong way"
            var_filter = raw_x.var(axis=0) > 0
            x = raw_x[:, var_filter]

        return x

    @classmethod
    def Y_features_prep(cls):

        ### Generate final y features in ndarray (-1, 1)
        cls.__dataloader_initializer()
        Y_labels = cls.synergy_score.loc[:, 'synergy']
        Y_half = Y_labels.values.reshape(-1, 1)
        Y = np.concatenate((Y_half, Y_half), axis=0)
        return Y

class SamplesDataLoader(CustomDataLoader):

    entrez_set = None
    cellline_set = None
    network = None
    simulated_drug_target = None
    synergy_score = None
    sel_dp = None
    expression_df = None
    netexpress_df = None
    drug_a_features = None
    drug_b_features = None
    cellline_features = None
    drug_features = None
    data_initialized = False
    whole_df = None
    Y = None
    drug_features_lengths = []
    cellline_features_lengths = []
    L1000_upregulation = None
    L1000_downregulation = None
    F_cl = None
    var_filter = None
    raw_x = None
    combine_drug_multi_gene_express = None
    single_drug_response = None

    def __init__(self):
        super().__init__()

    @classmethod
    def __dataloader_initializer(cls):

        if cls.data_initialized:
            return
        cls.entrez_set = GenesDataReader.get_gene_entrez_set()

        ### Reading network data
        ### entrez_a entrez_b association
        ### 1001 10001 0.3
        ### 10001 100001 0.2
        cls.network = NetworkDataReader.get_network()

        ### Creating test drug target matrix ###
        ###                   1001    10001    235       32          25           2222
        ### 5-FU               1        0        0        0           0            0
        ### ABT-888            0        0        0        0           0            0
        ### AZD1775            0        1        1        0           1            0
        ### BORTEZOMIB         1        1        1        1           0            1
        ### CARBOPLATIN        0        0        0        0           1            0
        if 'drug_target_profile' in setting.drug_features:
            cls.simulated_drug_target = DrugTargetProfileDataLoader.get_filtered_simulated_drug_target_matrix()

        ### Reading synergy score data ###
        ### Unnamed: 0,drug_a_name,drug_b_name,cell_line,synergy
        ### 5-FU_ABT-888_A2058,5-FU,ABT-888,A2058,7.6935301658
        ### 5-FU_ABT-888_A2780,5-FU,ABT-888,A2780,7.7780530601
        cls.synergy_score = SynergyDataReader.get_synergy_score()
        cls.cellline_set = SynergyDataReader.get_synergy_data_cell_lines()

        ### Processing gene dependencies map
        ###     "X127399","X1321N1","X143B",
        ### entrez
        ### 1001
        ### 10001
        if 'gene_dependence' in setting.cellline_features:
            cls.sel_dp = GeneDependenciesDataReader.get_gene_dp()

        ### Prepare gene expression data information
        cls.expression_df = ExpressionDataLoader.prepare_expresstion_df(entrezIDs=list(cls.entrez_set),
                                                                            celllines=list(cls.cellline_set))

        if 'netexpress' in setting.cellline_features:
            cls.netexpress_df = NetExpressDataLoader.prepare_netexpress_df(entrezIDs=list(cls.entrez_set))

        if setting.add_single_response_to_drug_target:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler(with_mean=False)
            cls.single_drug_response = pd.read_csv(setting.single_response, index_col=0)
            cls.single_drug_response['pIC50'] = scaler.fit_transform(cls.single_drug_response[['pIC50']]).reshape(-1,1)
            if 'drug' in cls.single_drug_response.columns:
                cls.single_drug_response['drug'] = cls.single_drug_response['drug'].str.upper()

        ######################
        ### 5-FU ....
        #####################
        if 'L1000_upregulation' in setting.drug_features:
            cls.L1000_upregulation = pd.read_csv(setting.L1000_upregulation, header = None, index_col = 0)
        if 'L1000_downregulation' in setting.drug_features:
            cls.L1000_downregulation = pd.read_csv(setting.L1000_downregulation, header = None, index_col = 0)

        if 'combine_drugs_for_cl' in setting.cellline_features:

            cls.combine_drug_multi_gene_express = \
            network_propagation.drug_combine_multiplication_gene_expression_network_propagation(cls.network,
                                                                                                cls.expression_df,
                                                                                                cls.entrez_set,
                                                                                                cls.simulated_drug_target,
                                                                                                cls.synergy_score,
                                                                                                setting.gene_expression_simulated_result_matrix)

        cls.__check_data_frames()
        cls.data_initialized = True



    @classmethod
    def __drug_features_prep(cls):


        ### generate drugs features
        if cls.drug_a_features is None or cls.drug_b_features is None or cls.drug_features is None:
            cls.__dataloader_initializer()
            cls.drug_features = []
            cls.drug_a_features = []
            cls.drug_b_features = []

            if 'drug_target_profile' in setting.drug_features:

                drug_a_target_feature = cls.simulated_drug_target.loc[list(cls.synergy_score['drug_a_name']), :]
                drug_a_target_feature = pd.DataFrame(drug_a_target_feature, columns=cls.entrez_set).reset_index(drop=True)
                if setting.add_single_response_to_drug_target:

                    # drug_a_single_response = cls.single_drug_response.loc[list(cls.synergy_score['drug_a_name']), :]

                    drug_a_single_response = cls.single_drug_response.merge(cls.synergy_score,
                                                                            left_on = ['drug', 'cell_line'],
                                                                            right_on = ['drug_a_name', 'cell_line'])['pIC50'].values
                    assert len(drug_a_single_response) == len(drug_a_target_feature), "single repsonse data didn't have same length with drug feature"
                    drug_a_target_feature['pIC50'] = drug_a_single_response

                drug_a_target_feature.fillna(0, inplace=True)
                cls.drug_a_features.append(drug_a_target_feature.values)
                cls.drug_features_lengths.append(drug_a_target_feature.shape[1])
                drug_b_target_feature = cls.simulated_drug_target.loc[list(cls.synergy_score['drug_b_name']), :]
                drug_b_target_feature = pd.DataFrame(drug_b_target_feature, columns=cls.entrez_set).reset_index(drop=True)
                if setting.add_single_response_to_drug_target:
                    # drug_b_single_response = cls.single_drug_response.loc[list(cls.synergy_score['drug_b_name']), :]
                    drug_b_single_response = cls.single_drug_response.merge(cls.synergy_score,
                                                                            left_on = ['drug', 'cell_line'],
                                                                            right_on = ['drug_b_name', 'cell_line'])['pIC50'].values
                    assert len(drug_b_single_response) == len(drug_b_target_feature), "single repsonse data didn't have same length with drug feature"

                    drug_b_target_feature['pIC50'] = drug_b_single_response
                drug_b_target_feature.fillna(0, inplace=True)
                cls.drug_b_features.append(drug_b_target_feature.values)

            if 'L1000_upregulation' in setting.drug_features:

                drug_a_F_feature = cls.L1000_upregulation.loc[list(cls.synergy_score['drug_a_name']), :]
                cls.drug_a_features.append(drug_a_F_feature.values)
                cls.drug_features_lengths.append(drug_a_F_feature.shape[1])
                drug_b_F_feature = cls.L1000_upregulation.loc[list(cls.synergy_score['drug_b_name']), :]
                cls.drug_b_features.append(drug_b_F_feature.values)

            if 'L1000_downregulation' in setting.drug_features:

                drug_a_F_feature_2 = cls.L1000_downregulation.loc[list(cls.synergy_score['drug_a_name']), :]
                cls.drug_a_features.append(drug_a_F_feature_2.values)
                cls.drug_features_lengths.append(drug_a_F_feature_2.shape[1])
                drug_b_F_feature_2 = cls.L1000_downregulation.loc[list(cls.synergy_score['drug_b_name']), :]
                cls.drug_b_features.append(drug_b_F_feature_2.values)

        return [cls.drug_a_features, cls.drug_b_features]

    @classmethod
    def __cellline_features_prep(cls):

        if cls.cellline_features is None:
            cls.__dataloader_initializer()
            cls.cellline_features = []
            dp_features = None
            ### generate cell lines features
            if 'gene_dependence' in setting.cellline_features:

                dp_features = cls.sel_dp[list(cls.synergy_score['cell_line'])].T
                dp_features = pd.DataFrame(dp_features, columns=cls.entrez_set).reset_index(drop=True)
                dp_features.fillna(0, inplace=True)
                cls.cellline_features.append(dp_features.values)
                cls.cellline_features_lengths.append(dp_features.shape[1])
            if 'combine_drugs_for_cl' in setting.cellline_features:

                combine_drug_multi_gene_express = pd.DataFrame(cls.combine_drug_multi_gene_express, columns=cls.entrez_set).reset_index(drop=True)
                combine_drug_multi_gene_express.fillna(0, inplace=True)
                if setting.expression_dependencies_interaction and dp_features is not None:
                    combine_drug_multi_gene_express = pd.DataFrame(np.multiply(combine_drug_multi_gene_express.values, dp_features.values),
                                                            index=combine_drug_multi_gene_express.index,
                                                            columns=combine_drug_multi_gene_express.columns)
                cls.cellline_features.append(combine_drug_multi_gene_express.values)
                cls.cellline_features_lengths.append(combine_drug_multi_gene_express.shape[1])

            if 'gene_expression' in setting.cellline_features:
                cellline_express_features = cls.expression_df.T.loc[list(cls.synergy_score['cell_line']), :]
                cls.cellline_features.append(cellline_express_features.values)
                cls.cellline_features_lengths.append(cellline_express_features.shape[1])

            if 'netexpress' in setting.cellline_features:
                netexpress_feature = cls.netexpress_df.T.loc[list(cls.synergy_score['cell_line']), :]
                cls.cellline_features.append(netexpress_feature.values)#.values/np.absolute(netexpress_feature.values).max())
                cls.cellline_features_lengths.append(netexpress_feature.shape[1])

            if setting.add_single_response_to_drug_target:


                for i in range(len(cls.cellline_features)):
                    # cell_line_single_response = cls.single_drug_response.loc[list(cls.synergy_score['cell_line']), :]
                    # assert len(cell_line_single_response) == len(cls.cellline_features[i]), "single repsonse data didn't have same length with cell line fetures"
                    # cls.cellline_features[i] = np.concatenate([cls.cellline_features[i], cell_line_single_response.values], axis = 1)

                    cls.cellline_features[i] = np.concatenate([cls.cellline_features[i],
                                                        np.array([[0] * len(cls.cellline_features[i])]).reshape(-1,1)],
                                                       axis=1)
                    cls.cellline_features_lengths[i] += 1

        return cls.cellline_features

    @classmethod
    def __construct_whole_raw_X(cls):

        ### return dataframe
        ###  first_half_drugs_features                first_half_cellline_features
        ###  switched_second_half_drugs_features      second_half_cellline_features
        if cls.whole_df is None:
            two_drugs_features_list = cls.__drug_features_prep()
            cellline_features_list = cls.__cellline_features_prep()
            first_half = np.concatenate(tuple(two_drugs_features_list[0] + two_drugs_features_list[1] +
                                              cellline_features_list), axis=1)
            second_half = np.concatenate(tuple(two_drugs_features_list[1] + two_drugs_features_list[0] +
                                               cellline_features_list), axis=1)
            cls.whole_df = np.concatenate(tuple([first_half, second_half]), axis=0)#.reset_index(drop=True)
        return cls.whole_df

    @classmethod
    def Raw_X_features_prep(cls, methods):

        ### Generate final raw features dataset
        ### return: ndarray (n_samples, n_type_features, feature_dim) if 'attn'
        ###         ndarray (n_samples, n_type_features * feature_dim) else
        raw_x = cls.__construct_whole_raw_X()
        entrez_array = np.array(list(cls.entrez_set))
        if methods == 'attn':
            x = raw_x.reshape(-1, setting.n_feature_type, len(cls.entrez_set))
            filter_drug_features_len = filter_cl_features_len = x.shape[-1]
            drug_features_name = cl_features_name = cls.entrez_set

        elif methods == 'flexible_attn':

            return raw_x, cls.drug_features_lengths, cls.cellline_features_lengths

        else:
            drug_features_len = int(1 / setting.n_feature_type * raw_x.shape[1])
            cl_features_len = int(raw_x.shape[1] - 2 * drug_features_len)
            assert cl_features_len == int((1 - 2 / setting.n_feature_type) * raw_x.shape[1]), \
                "features len are calculated in wrong way"
            var_filter = raw_x.var(axis=0) > 0
            filter_drug_features_len = sum(var_filter[:drug_features_len])
            filter_cl_features_len = sum(var_filter[2*drug_features_len:])
            drug_features_name = entrez_array[var_filter[:drug_features_len]]
            cl_features_name = np.array(list(entrez_array) * (setting.n_feature_type - 2))[var_filter[2 * drug_features_len:]]
            x = raw_x[:, var_filter]
            assert filter_drug_features_len == len(drug_features_name) and filter_cl_features_len == len(cl_features_name), \
                                                                                  'features len and names do not match'
        return x, filter_drug_features_len, filter_cl_features_len, list(drug_features_name), list(cl_features_name)

    @classmethod
    def Y_features_prep(cls):

        ### Generate final y features in ndarray (-1, 1)
        #cls.__dataloader_initializer()
        if cls.synergy_score is None:
            cls.synergy_score = SynergyDataReader.get_synergy_score()
        Y_labels = cls.synergy_score.loc[:, 'synergy']
        Y_half = Y_labels.values.reshape(-1, 1)
        Y = np.concatenate((Y_half, Y_half), axis=0)
        return Y

    @classmethod
    def __check_data_frames(cls):

        if 'drug_target_profile' in setting.drug_features:
            logger.debug("check_unfound_genes_in_drug_target ...")
            DrugTargetProfileDataLoader.check_unfound_genes_in_drug_target()
            logger.debug("check_drugs_in_drug_target ... ")
            DrugTargetProfileDataLoader.check_drugs_in_drug_target()
        if 'gene_dependence' in setting.cellline_features:
            logger.debug("check_unfound_genes_in_gene_dp ... ")
            GeneDependenciesDataReader.check_unfound_genes_in_gene_dp()
            logger.debug("check_celllines_in_gene_dp...")
            GeneDependenciesDataReader.check_celllines_in_gene_dp()
        logger.debug("check_genes_in_network ...")
        NetworkDataReader.check_genes_in_network()


        ### select only the drugs with features
        ### select only the drug targets in genes

class DataPreprocessor:

    X = None
    Y = None
    drug_features_len = None
    cl_features_len = None
    synergy_score = None
    methods = None

    def __init__(self, methods):
        self.methods = methods
        pass

    @classmethod
    def __dataset_initializer(cls):

        if cls.X is None:
            cls.X, cls.drug_features_len, cls.cl_features_len, _, _ = SamplesDataLoader.Raw_X_features_prep(cls.methods)
        if cls.Y is None:
            cls.Y = SamplesDataLoader.Y_features_prep()
        if cls.synergy_score is None:
            cls.synergy_score = SynergyDataReader.get_synergy_score()

    @classmethod
    def reg_train_eval_test_split(cls, fold = 'fold', test_fold = 0):

        if cls.synergy_score is None:
            cls.synergy_score = SynergyDataReader.get_synergy_score()

        if setting.index_in_literature:
            evluation_fold = np.random.choice(list({0,1,2,3,4}-{test_fold}))
            evluation_fold = 0
            print(evluation_fold)
            test_index = np.array(cls.synergy_score[cls.synergy_score[fold] == test_fold].index)
            evaluation_index = np.array(cls.synergy_score[cls.synergy_score[fold] == evluation_fold].index)
            train_index = np.array(cls.synergy_score[(cls.synergy_score[fold] != test_fold) &
                                                     (cls.synergy_score[fold] != evluation_fold)].index)

        else:
            train_index, test_index = drug_drug.split_data(cls.synergy_score, group_df=cls.synergy_score, group_col=[fold])
            train_index, evaluation_index = drug_drug.split_data(cls.synergy_score,
                                                                 group_df=cls.synergy_score[train_index],
                                                                 group_col=[fold])

        train_index = np.concatenate([train_index, train_index + cls.synergy_score.shape[0]])
        evaluation_index_2 = evaluation_index + cls.synergy_score.shape[0]
        test_index_2 = test_index + cls.synergy_score.shape[0]
        if setting.unit_test:
            train_index, test_index, test_index_2, evaluation_index, evaluation_index_2 = \
                train_index[:100], test_index[:100], test_index_2[:100], evaluation_index[:100], evaluation_index_2[:100]
        yield train_index, test_index, test_index_2, evaluation_index, evaluation_index_2

    @classmethod
    def cv_train_eval_test_split_generator(cls, fold = 'fold'):

        if cls.synergy_score is None:
            cls.synergy_score = SynergyDataReader.get_synergy_score()

        assert setting.index_in_literature, "Cross validation is only available when index_in_literature is set to True"
        for evluation_fold in range(1,5):
            test_index = np.array(cls.synergy_score[cls.synergy_score[fold] == 0].index)
            evaluation_index = np.array(cls.synergy_score[cls.synergy_score[fold] == evluation_fold].index)
            train_index = np.array(cls.synergy_score[(cls.synergy_score[fold] != 0) &
                                                     (cls.synergy_score[fold] != evluation_fold)].index)
            train_index = np.concatenate([train_index + cls.synergy_score.shape[0], train_index])
            evaluation_index_2 = evaluation_index + cls.synergy_score.shape[0]
            test_index_2 = test_index + cls.synergy_score.shape[0]
            if setting.unit_test:
                train_index, test_index, test_index_2, evaluation_index, evaluation_index_2 = \
                    train_index[:100], test_index[:100], test_index_2[:100], evaluation_index[:100], evaluation_index_2[:100]
            yield train_index, test_index, test_index_2, evaluation_index, evaluation_index_2

class MyDataset(data.Dataset):

    synergy_score = None
    drug_smile = None
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, prefix=None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.prefix = prefix
        if MyDataset.synergy_score is None:
            print('prepare synergy score')
            MyDataset.synergy_score = SynergyDataReader.get_synergy_score()
            synergy_score_reverse = MyDataset.synergy_score.copy()
            synergy_score_reverse['drug_a_name'] = MyDataset.synergy_score['drug_b_name']
            synergy_score_reverse['drug_b_name'] = MyDataset.synergy_score['drug_a_name']
            MyDataset.synergy_score = pd.concat([MyDataset.synergy_score, synergy_score_reverse])
            MyDataset.synergy_score.reset_index(inplace=True)
        if MyDataset.drug_smile is None:
            print('prepare drug smile')
            name_smile_df = pd.read_csv(setting.inchi_merck)
            MyDataset.drug_smile = {name: smile for name, smile in zip(name_smile_df['Name'], name_smile_df['SMILE'])}

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        if self.prefix is None:
            drug_combine_file = path.join(setting.data_folder, ID + '.pt')
        else:
            drug_combine_file = self.prefix + '_datas/' + ID + '.pt'
        # Load data and get label
        try:
            X = torch.load(drug_combine_file)
        except:
            logger.error("Fail to get {}".format(ID))
            raise
        y = self.labels[ID]
        drug_a = MyDataset.synergy_score.loc[index, 'drug_a_name']
        drug_a_smiles = MyDataset.drug_smile[drug_a]
        drug_b = MyDataset.synergy_score.loc[index, 'drug_b_name']
        drug_b_smiles = MyDataset.drug_smile[drug_b]

        return (X, drug_a_smiles, drug_b_smiles), y
