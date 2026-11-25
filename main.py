import argparse
from model_pipeline import (
    evaluate_model,
    load_model,
    predict,
    prepare_data,
    save_model,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Churn ML Pipeline CLI")
    parser.add_argument("--all", action="store_true", help="Run all steps in sequence")
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", action="store_true", help="Run predictions")
    parser.add_argument("--save", action="store_true", help="Train and save model")
    parser.add_argument("--load", action="store_true", help="Load saved model")
    args = parser.parse_args()

    if args.all:
        x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
        loaded = load_model()
        evaluate_model(loaded, x_test, y_test)
        preds = predict(x_test)
        print(f"Predictions: {len(preds)}")
        return

    if args.prepare:
        prepare_data()
    if args.train:
        x_train, _, y_train, _ = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
    if args.evaluate:
        _, x_test, _, y_test = prepare_data()
        model = load_model()
        evaluate_model(model, x_test, y_test)
    if args.predict:
        _, x_test, _, _ = prepare_data()
        preds = predict(x_test)
        print(f"Predictions: {len(preds)}")
    if args.save:
        x_train, _, y_train, _ = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
    if args.load:
        load_model()


if __name__ == "__main__":
    main()
