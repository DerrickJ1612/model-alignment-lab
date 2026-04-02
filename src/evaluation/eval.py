import json 
import pandas as pd

from src.utils.helpers import generate_response

def log_parser(trainer):
    """
    Extracts training and evaluation metrics from a Hugging Face Trainer.

    This function parses the trainer's `state.log_history` to collect:
    - Epoch numbers
    - Training loss values
    - Evaluation loss values

    Only logs at whole-number epochs (e.g., 1.0, 2.0, ...) are included.

    Args:
        trainer: Hugging Face Trainer object after or during training.

    Returns:
        tuple:
            - epoch (List[float]): Epoch values corresponding to logged training steps
            - loss (List[float]): Training loss values
            - eval_loss (List[float]): Evaluation loss values
    """
    epoch, loss, eval_loss = [], [], []

    for log in trainer.state.log_history:
        if "epoch" in log and log["epoch"] % 1.0 == 0:
            if "loss" in log:
                epoch.append(log["epoch"])
                loss.append(log["loss"])
            elif "eval_loss" in log:
                eval_loss.append(log["eval_loss"])
            else:
                continue

    return (epoch, loss, eval_loss)

def evaluate_to_dataframe(model, tokenizer, dataset):
    """
    Runs inference on a dataset and compares predicted tool outputs against ground truth.

    For each sample:
    - Extracts the user prompt and reference response
    - Generates a model prediction using `generate_tool`
    - Attempts to parse both prediction and reference as JSON
    - Compares:
        - Tool name
        - Arguments
        - JSON validity

    The results are returned as a Pandas DataFrame for analysis.

    Expected dataset format:
        Each sample must contain:
        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}  # JSON string
            ]
        }

    Args:
        model: The language model used for inference.
        tokenizer: Corresponding tokenizer.
        dataset: Iterable dataset (e.g., Hugging Face Dataset).

    Returns:
        pd.DataFrame:
            Columns include:
                - prompt
                - pred_tool / actual_tool
                - tool_match (bool)
                - pred_arguments / actual_arguments
                - arguments_match (bool)
                - valid_json (bool)
                - prediction_raw
                - reference_raw
    """
    rows = []
    for i, sample in enumerate(dataset):
        prompt = sample["messages"][0]["content"]
        reference = sample["messages"][1]["content"]

        pred = generate_response(model, tokenizer, prompt)

        # defaults
        pred_tool = None
        pred_args = None
        actual_tool = None
        actual_args = None
        valid_json = False

        try:
            pred_json = json.loads(pred)
            valid_json = True
            pred_tool = pred_json.get("tool")
            pred_args = pred_json.get("arguments")
        except json.JSONDecodeError:
            pass

        try:
            ref_json = json.loads(reference)
            actual_tool = ref_json.get("tool")
            actual_args = ref_json.get("arguments")
        except json.JSONDecodeError:
            pass

        rows.append({
            "prompt": prompt,
            "pred_tool": pred_tool,
            "actual_tool": actual_tool,
            "tool_match": pred_tool == actual_tool,
            "pred_arguments": pred_args,
            "actual_arguments": actual_args,
            "arguments_match": pred_args == actual_args,
            "valid_json": valid_json,
            "prediction_raw": pred,
            "reference_raw": reference
        })

    return pd.DataFrame(rows)
