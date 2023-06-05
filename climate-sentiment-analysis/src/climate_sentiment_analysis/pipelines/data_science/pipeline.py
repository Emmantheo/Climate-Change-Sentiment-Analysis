"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from climate_sentiment_analysis.pipelines.data_science.nodes import split_data, vectorize_text, train_model
from climate_sentiment_analysis.pipelines.data_science.nodes import evaluate_model,vectorize_new_text, get_predictions

def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=split_data,
                inputs=["cleaned_train_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=vectorize_text,
                inputs=["X_train", "X_test","parameters"],
                outputs=['X_train_vec', 'X_test_vec', "vectorizer"],
                name="vectorize_text_node",
            ),
            node(
                func=train_model,
                inputs=["X_train_vec", "y_train", "parameters"],
                outputs="tweet_classifier",
                name="tweet_classifier_node",
            ),
            node(
                func=evaluate_model,
                inputs=["tweet_classifier", "X_test_vec", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
            node(
                func=vectorize_new_text,
                inputs=["cleaned_new_data", "vectorizer"],
                outputs='X_news_vec',
                name="vectorize_new_tweet_node",
            ),
            node(
                func=get_predictions,
                inputs=['cleaned_new_data','tweet_classifier', 'X_news_vec'],
                outputs='predictions',
                name="predict_new_tweet_node",
            ),
        ]
    )
    data_science = pipeline(
        pipe=pipeline_instance,
        inputs=["cleaned_train_data","cleaned_new_data"],
        namespace = "data_science",
        outputs = 'predictions'
    )
    return data_science