import torch
import numpy as np
import pandas as pd
import logging
from src import model, drug_drug, setting, my_data
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_drugs_profiles(raw_chemicals, genes):

    if not setting.drug_profiles_renew and os.path.exists(setting.drug_profiles):
        drug_profile = pd.read_csv(setting.drug_profiles, index_col=0)
        return drug_profile

    drug_profile = np.zeros(shape=(len(raw_chemicals), len(genes)))
    drug_profile = pd.DataFrame(drug_profile, columns=genes['entrez'], index=raw_chemicals['Name'])
    entrez_set = set(genes['entrez'])
    for row in raw_chemicals.iterrows():

        if not isinstance(row[1]['combin_entrez'], str):
            continue

        chem_name, target_list = row[1]['Name'], row[1]['combin_entrez'].split(",")
        for target in target_list:
            target = int(target)
            if target in entrez_set:
                drug_profile.loc[chem_name, target] = 1
    print(setting.drug_profiles)
    drug_profile.T.to_csv(setting.drug_profiles)
    return drug_profile.T

def train_model(drug_model, X_train, Y_train, X_val, Y_val):
    # Fetch training parameters from settings.py
    n_epochs = setting.n_epochs
    batch_size = setting.batch_size
    start_lr = setting.start_lr
    dropout = setting.dropout
    loss_fn = torch.nn.MSELoss()  # You can replace this with any loss function from settings if needed
    optimizer = torch.optim.Adam(drug_model.parameters(), lr=start_lr)

    # Convert the data to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

    # Move model to device (GPU if available)
    drug_model.to(device)
    X_train_tensor, Y_train_tensor = X_train_tensor.to(device), Y_train_tensor.to(device)
    X_val_tensor, Y_val_tensor = X_val_tensor.to(device), Y_val_tensor.to(device)

    # Training loop
    for epoch in range(n_epochs):
        drug_model.train()  # Set model to training mode
        optimizer.zero_grad()  # Zero the gradients

        # Iterate over batches
        for i in range(0, len(X_train), batch_size):
            # Get the current batch
            batch_X = X_train_tensor[i:i+batch_size]
            batch_Y = Y_train_tensor[i:i+batch_size]

            # Forward pass
            predictions = drug_model(batch_X)
            loss = loss_fn(predictions, batch_Y)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()  # Zero gradients for next batch

        # Validation step (after completing a full epoch)
        drug_model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need to compute gradients for validation
            val_predictions = drug_model(X_val_tensor)
            val_loss = loss_fn(val_predictions, Y_val_tensor)

        # Print the progress
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    return drug_model

if __name__ == "__main__":

    # Setting up nvidia GPU environment
    if not setting.ml_train:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Setting up log file
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh = logging.FileHandler(setting.logfile, mode='w+')
    fh.setFormatter(fmt=formatter)
    logger = logging.getLogger("Drug Combination")
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)

    std_scaler = StandardScaler()
    logger.debug("Getting features and synergy scores ...")
    X, drug_features_len, cl_features_len, drug_features_name, cl_features_name = \
        my_data.SamplesDataLoader.Raw_X_features_prep(methods='mlp')
    Y = my_data.SamplesDataLoader.Y_features_prep()
    logger.debug("Spliting data ...")

    cv_pearsonr_scores = []
    cvmodels = []
    best_test_index, best_test_index_2 = None, None
    for train_index, test_index, test_index_2, evaluation_index, evaluation_index_2 in\
        my_data.DataPreprocessor.cv_train_eval_test_split_generator():

        logger.debug("Splitted data successfully")
        std_scaler.fit(Y[train_index])
        if setting.y_transform:
            Y = std_scaler.transform(Y) * 100
            best_test_index, best_test_index_2 = test_index, test_index_2

        if setting.ml_train:

            x_cols = [x + "_a" for x in drug_features_name] + [x + "_b" for x in drug_features_name] + cl_features_name
            X = pd.DataFrame(X, columns=x_cols)
            Y = pd.DataFrame(Y, columns=['synergy'])
            drug_drug.__ml_train(X, Y, train_index, test_index)

        else:

            # Creating and compiling the model using the new `DrugsCombModel` class
            drug_model = model.DrugsCombModel(drug_a_features_len=drug_features_len,
                                              drug_b_features_len=drug_features_len,
                                              cl_features_len=cl_features_len).get_model()

            logger.info("model information: \n %s" % drug_model)
            logger.debug("Start training")

            drug_model = train_model(
                    drug_model,
                    X[train_index], Y[train_index],
                    X[test_index], Y[test_index]
                )

            logger.debug("Training is done")
            logger.debug("Start evaluation")
            drug_model.eval()

            # Disable gradient calculation for inference
            with torch.no_grad():
                X_train_tensor = torch.tensor(X[train_index], dtype=torch.float32).to(device)
                train_prediction = drug_model(X_train_tensor).reshape((-1,))
                
            train_prediction = std_scaler.inverse_transform(train_prediction / 100)
            Y = std_scaler.inverse_transform(Y / 100)
            train_mse = mean_squared_error(Y[train_index], train_prediction)
            train_pearson = pearsonr(Y[train_index].reshape(-1), train_prediction.reshape(-1))[0]

            logger.info("training dataset: mse: %s, pearson: %s" % (str(train_mse), str(1 - train_pearson ** 2)))
            
            with torch.no_grad():
                X_eval_tensor = torch.tensor(X[evaluation_index], dtype=torch.float32).to(device)
                eval_prediction = drug_model(X_eval_tensor).reshape((-1,))
                
            eval_prediction = std_scaler.inverse_transform(eval_prediction / 100)
            
            with torch.no_grad():
                X_eval2_tensor = torch.tensor(X[evaluation_index_2], dtype=torch.float32).to(device)
                eval_prediction_2 = drug_model(X_eval2_tensor).reshape((-1,))
            
            eval_prediction_2 = std_scaler.inverse_transform(eval_prediction_2 / 100)
            final_prediction = np.mean([eval_prediction, eval_prediction_2], axis=0)
            comparison = pd.DataFrame(
                {'ground_truth': Y[evaluation_index].reshape(-1), 'prediction': final_prediction.reshape(-1)})
            eval_mse = mean_squared_error(Y[evaluation_index], final_prediction)
            eval_pearson = pearsonr(Y[evaluation_index].reshape(-1), final_prediction.reshape(-1))[0]
            cv_pearsonr_scores.append(eval_pearson)

            logger.info("Evaluation dataset: mse: %s, pearson: %s" % (str(eval_mse), str(1 - eval_pearson ** 2)))

            cvmodels.append(drug_model)

    best_index = 0
    for i in range(len(cv_pearsonr_scores)):
        if cv_pearsonr_scores[best_index] < cv_pearsonr_scores[i]:
            best_index = i

    best_model = cvmodels[best_index]
    logger.info("Best model index: %s" % str(best_index))
    logger.info("Best model pearson: %s" % str(cv_pearsonr_scores[best_index]))
    logger.info("Best model mse: %s" % str(1 - cv_pearsonr_scores[best_index] ** 2))
    
    logger.info("Start testing")
    best_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X[best_test_index], dtype=torch.float32).to(device)
        test_prediction = best_model(X_test_tensor).reshape((-1,))
    test_prediction = std_scaler.inverse_transform(test_prediction / 100)
    
    with torch.no_grad():
        X_test2_tensor = torch.tensor(X[best_test_index_2], dtype=torch.float32).to(device)
        test_prediction_2 = best_model(X_test2_tensor).reshape((-1,))
        
    test_prediction_2 = std_scaler.inverse_transform(test_prediction_2 / 100)
    
    final_prediction = np.mean([test_prediction, test_prediction_2], axis=0)
    comparison = pd.DataFrame({'ground_truth': Y[best_test_index].reshape(-1), 'prediction': final_prediction.reshape(-1)})
    comparison.to_csv("last_output_{!r}".format(int(time())) + ".csv")
    test_mse = mean_squared_error(Y[best_test_index], final_prediction)
    test_pearson = pearsonr(Y[best_test_index].reshape(-1), final_prediction.reshape(-1))[0]

    logger.info("Testing dataset: mse: %s, pearson: %s" % (str(test_mse), str(1 - test_pearson ** 2)))
