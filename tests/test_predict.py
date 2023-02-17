from jc_thesis_code import data, predict


def test_predict_one_with_pkg_model(large_df):
    predictionHandler = predict.load_handler_with_pkg_model()
    (
        defect,
        proba_prediction,
        preprocess_duration,
        predict_duration,
    ) = predictionHandler.predict_one(large_df)
    assert isinstance(defect, data.Defect)
    assert all([proba_prediction[defect] >= proba for proba in proba_prediction])
    assert isinstance(preprocess_duration, float)
    assert isinstance(predict_duration, float)
