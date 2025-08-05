from transformers import pipeline
from transformers.pipelines.base import Pipeline
import logging

logger = logging.getLogger("models")

def generate_category_zero_shot_classifer(category_classifier_model: str, device: str = "cpu") -> Pipeline:
    """
    Takes a model string and a device string and returns an instantiated pipeline
    of a zero shot classifier for simple categories

    Parameters:
    category_classifier_model (str): The name of the model as on huggingface to be used in the pipeline
    device (str): The device to use for the model

    Returns:
    classifier (transformers.base.pipelines.Pipeline): The complete category classifier pipeline
    """
    classifier = pipeline("zero-shot-classification", model=category_classifier_model, device=device)
    logger.debug(f"Instantiated category classifier pipeline")
    return classifier

def generate_emotion_classifier(emotion_classifer_model: str, device: str = "cpu") -> Pipeline:
    """
    Takes a model string and a device string and returns an instantiated pipeline
    of a fine-tuned classifier for emotions

    Parameters:
    emotion_classifier_model (str): The name of the model as on huggingface to be used in the pipeline
    device (str): The device to use for the model

    Returns:
    classifier (transformers.base.pipelines.Pipeline): The complete emotion classifier pipeline
    """
    classifier = pipeline(
        "text-classification",
        model=emotion_classifer_model,
        device=device,
        top_k = None,
        truncation=True,
        max_length=512 # Model's limit is 512 tokens
    )
    logger.debug(f"Instantiated emotion classifier pipeline")
    return classifier