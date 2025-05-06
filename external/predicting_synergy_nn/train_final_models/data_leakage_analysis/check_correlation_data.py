import pandas as pd
import matplotlib.pyplot as plt
import os

for fold in range(1, 4):       
    folder = f"../train_fold{fold}"
    train_file = os.path.join(folder, f"fold{fold}_alltrain.csv")
    valid_file = os.path.join(folder, f"fold{fold}_test.csv")

    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        print(f"Skipping {folder} due to missing files.")
        continue

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(valid_file)

    X_train = train_data.iloc[:, :33] 
    y_train = train_data['ZIP']  
    X_test = test_data.iloc[:, :33]  
    y_test = test_data['ZIP'] 

    correlation_train = X_train.corrwith(y_train)
    correlation_test = X_test.corrwith(y_test)

    leakage_train = correlation_train[correlation_train.abs() > 0.9]  
    leakage_test = correlation_test[correlation_test.abs() > 0.9] 

    if not leakage_train.empty:
        print("Potential data leakage in training set (correlations > 0.9):")
        print(leakage_train)

    if not leakage_test.empty:
        print("\nPotential data leakage in testing set (correlations > 0.9):")
        print(leakage_test)

    if not leakage_train.empty:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Training Set Feature Correlations with Target")
        leakage_train.plot(kind='bar')

    if not leakage_test.empty:
        plt.subplot(1, 2, 2)
        plt.title("Testing Set Feature Correlations with Target")
        leakage_test.plot(kind='bar')

    if not leakage_train.empty or not leakage_test.empty:
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo significant correlations between features and target detected, correlations < 0.9")
