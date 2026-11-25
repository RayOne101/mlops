import argparse
import subprocess
from prefect import flow, task
from model_pipeline import (
    evaluate_model,
    load_model,
    predict,
    prepare_data,
    save_model,
    train_model,
)

FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


@task
def install_deps():
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)


@task
def format_code():
    subprocess.run(["black", *FILES], check=True)


@task
def code_quality():
    subprocess.run(["flake8", *FILES], check=True)


@task
def code_security():
    subprocess.run(["bandit", "-ll", *FILES], check=True)


@task
def unit_tests():
    subprocess.run(["pytest"], check=True)


@task
def prep():
    return prepare_data()


@task
def train(x_train, y_train):
    return train_model(x_train, y_train)


@task
def save(model):
    save_model(model)


@task
def load():
    return load_model()


@task
def evaluate(model, x_test, y_test):
    return evaluate_model(model, x_test, y_test)


@task
def run_predict(x_test):
    preds = predict(x_test)
    print(f"Predictions: {len(preds)}")
    return preds


@flow(name="code")
def code_flow():
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="train")
def train_flow():
    x_train, _, y_train, _ = prep()
    model = train(x_train, y_train)
    save(model)


@flow(name="predict")
def predict_flow():
    _, x_test, _, _ = prep()
    run_predict(x_test)


@flow(name="all")
def all_flow():
    install_deps()
    code_flow()
    x_train, x_test, y_train, y_test = prep()
    model = train(x_train, y_train)
    save(model)
    loaded = load()
    evaluate(loaded, x_test, y_test)
    run_predict(x_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", required=True)
    args = parser.parse_args()

    if args.flow == "all":
        all_flow()
    elif args.flow in ("train", "entrainement"):
        train_flow()
    elif args.flow == "predict":
        predict_flow()
    elif args.flow == "code":
        code_flow()
