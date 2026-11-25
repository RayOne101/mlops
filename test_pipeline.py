import os
import model_pipeline


def test_prepare_data():
    x_train, x_test, y_train, y_test = model_pipeline.prepare_data()
    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert os.path.exists("scaler.joblib")


def test_train_model():
    x_train, _, y_train, _ = model_pipeline.prepare_data()
    model = model_pipeline.train_model(x_train, y_train)
    assert model is not None


def test_save_and_load_model():
    x_train, _, y_train, _ = model_pipeline.prepare_data()
    model = model_pipeline.train_model(x_train, y_train)
    model_pipeline.save_model(model, "test_model.joblib")
    loaded = model_pipeline.load_model("test_model.joblib")
    assert loaded is not None
    os.remove("test_model.joblib")


def test_evaluate_model():
    x_train, x_test, y_train, y_test = model_pipeline.prepare_data()
    model = model_pipeline.train_model(x_train, y_train)
    acc = model_pipeline.evaluate_model(model, x_test, y_test)
    assert 0.0 <= acc <= 1.0


def test_predict():
    x_train, x_test, y_train, _ = model_pipeline.prepare_data()
    model = model_pipeline.train_model(x_train, y_train)
    model_pipeline.save_model(model)
    preds = model_pipeline.predict(x_test[:5])
    assert len(preds) == 5
