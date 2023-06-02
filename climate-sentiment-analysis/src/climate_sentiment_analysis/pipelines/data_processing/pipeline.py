"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline
from climate_sentiment_analysis.pipelines.data_processing.nodes import preprocess_tweets, preprocess_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])

def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        [node(
            func=preprocess_tweets,
            inputs="train_data",
            outputs="cleaned_train_data",
            name="train_preprocess_data_node"
        ),
        node(
            func=preprocess_test,
            inputs="new_data",
            outputs="cleaned_new_data",
            name="test_preprocess_data_node"
        )
        ]
    )
    data_processing = pipeline(
        pipe=pipeline_instance,
        inputs=["train_data", "new_data"],
        namespace = "data_processing",
        outputs = ["cleaned_train_data", "cleaned_new_data"]
    )
    return data_processing
