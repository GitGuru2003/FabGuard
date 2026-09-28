"""Train all models with stratified CV on the training split and save them to Models/.

Usage: python Scripts/train.py
"""

from FabGuard.data import load_secom, split_data


def main():
    X, y, _ = load_secom()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {len(y_train)} samples ({y_train.sum()} failures)")
    print(f"Test:  {len(y_test)} samples ({y_test.sum()} failures)")
    # TODO: build pipelines, tune hyperparameters with StratifiedKFold, save models


if __name__ == "__main__":
    main()
