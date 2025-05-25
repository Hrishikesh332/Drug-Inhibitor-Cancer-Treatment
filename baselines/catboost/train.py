from sklearn.metrics import mean_squared_error

from external.predicting_synergy_nn.src.utils.data_loader import CVDatasetHandler
import catboost
import joblib
from scipy.stats import pearsonr, spearmanr

BEST_PARAMS = {'learning_rate': 0.06, 'l2_leaf_reg': 4, 'depth': 9, 'boosting_type': 'Plain'}

def train_and_eval_biomining_catboost(
    biomining_data_dir_path: str = "external/predicting_synergy_nn/data",
):
    data_loader = CVDatasetHandler(
        data_dir=biomining_data_dir_path
    )

    model = catboost.CatBoostRegressor(**BEST_PARAMS)

    X_train, y_train = data_loader.get_dataset(type="alltrain")
    X_test, y_test = data_loader.get_dataset(type="test")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    spearman = spearmanr(y_test, y_pred)[0]
    pearson = pearsonr(y_test, y_pred)[0]
    print(f"Mean Squared Error on test set: {mse:.4f}")
    print(f"Pearson on test set: {pearson:.4f}")
    print(f"Spearman on test set: {spearman:.4f}")

    return model

def train_eval_save_biomining_catboost(
        biomining_data_dir_path: str = "external/predicting_synergy_nn/data",
        model_output_path: str = "baselines/catboost/models/best_catboost_biomining.pkl"
):
    model = train_and_eval_biomining_catboost(biomining_data_dir_path)
    joblib.dump(model, model_output_path)

if __name__ == "__main__":
    train_eval_save_biomining_catboost()
