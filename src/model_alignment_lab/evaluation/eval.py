import json 
import pandas as pd

from model_alignment_lab.utils.helpers import generate_response

def log_parser(trainer):
    """
    Extract training and evaluation metrics from Hugging Face Trainer logs.

    For small LoRA experiments, plotting all logged training points is
    usually more informative than restricting to integer epochs.

    Args:
        trainer: Hugging Face Trainer object

    Returns:
        tuple:
            - train_epochs (List[float])
            - train_losses (List[float])
            - eval_epochs (List[float])
            - eval_losses (List[float])
    """

    train_epochs = []
    train_losses = []

    eval_epochs = []
    eval_losses = []

    for log in trainer.state.log_history:

        # Training logs
        if "loss" in log and "epoch" in log:
            train_epochs.append(log["epoch"])
            train_losses.append(log["loss"])

        # Evaluation logs
        if "eval_loss" in log and "epoch" in log:
            eval_epochs.append(log["epoch"])
            eval_losses.append(log["eval_loss"])

    return (train_epochs, train_losses, eval_epochs, eval_losses)

def evaluate_tool_calling_df(model, tokenizer, dataset):
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

def evaluate_tutor_schema_df(model, tokenizer, dataset):
    """
    Evaluates structured electromagnetics tutor outputs.

    Expected schema:
    {
        "problem_type": str,
        "difficulty": str,
        "required_concepts": list[str],
        "solution_strategy": list[str],
        "key_equations": list[str],
        "final_answer": dict,
        "common_mistakes": list[str]
    }

    Returns:
        pd.DataFrame containing:
            - schema field matches
            - JSON validity
            - overall structured accuracy
            - raw prediction/reference
    """

    rows = []

    for sample in dataset:

        prompt = sample["messages"][0]["content"]
        reference = sample["messages"][1]["content"]

        pred = generate_response(model, tokenizer, prompt)

        # ----------------------------------------
        # Defaults
        # ----------------------------------------

        valid_json = False

        pred_json = {}
        ref_json = {}

        # ----------------------------------------
        # Parse prediction JSON
        # ----------------------------------------

        try:
            pred_json = json.loads(pred)
            valid_json = True

        except json.JSONDecodeError:
            pred_json = {}

        # ----------------------------------------
        # Parse reference JSON
        # ----------------------------------------

        try:
            ref_json = json.loads(reference)

        except json.JSONDecodeError:
            ref_json = {}

        # ----------------------------------------
        # Extract prediction fields
        # ----------------------------------------

        pred_problem_type = pred_json.get(
            "problem_type"
        )

        pred_difficulty = pred_json.get(
            "difficulty"
        )

        pred_required_concepts = pred_json.get(
            "required_concepts", []
        )

        pred_solution_strategy = pred_json.get(
            "solution_strategy", []
        )

        pred_key_equations = pred_json.get(
            "key_equations", []
        )

        pred_final_answer = pred_json.get(
            "final_answer"
        )

        pred_common_mistakes = pred_json.get(
            "common_mistakes", []
        )

        # ----------------------------------------
        # Extract reference fields
        # ----------------------------------------

        actual_problem_type = ref_json.get(
            "problem_type"
        )

        actual_difficulty = ref_json.get(
            "difficulty"
        )

        actual_required_concepts = ref_json.get(
            "required_concepts", []
        )

        actual_solution_strategy = ref_json.get(
            "solution_strategy", []
        )

        actual_key_equations = ref_json.get(
            "key_equations", []
        )

        actual_final_answer = ref_json.get(
            "final_answer"
        )

        actual_common_mistakes = ref_json.get(
            "common_mistakes", []
        )

        # ----------------------------------------
        # Matching Metrics
        # ----------------------------------------

        problem_type_match = (
            pred_problem_type == actual_problem_type
        )

        difficulty_match = (
            pred_difficulty == actual_difficulty
        )

        concepts_match = (
            set(pred_required_concepts)
            == set(actual_required_concepts)
        )

        strategy_match = (
            set(pred_solution_strategy)
            == set(actual_solution_strategy)
        )

        equations_match = (
            set(pred_key_equations)
            == set(actual_key_equations)
        )

        final_answer_match = (
            pred_final_answer == actual_final_answer
        )

        mistakes_match = (
            set(pred_common_mistakes)
            == set(actual_common_mistakes)
        )

        overall_match = all([
            valid_json,
            problem_type_match,
            difficulty_match,
            concepts_match,
            strategy_match,
            equations_match,
            final_answer_match,
            mistakes_match
        ])

        # ----------------------------------------
        # Store row
        # ----------------------------------------

        rows.append({

            "prompt": prompt,

            "valid_json": valid_json,

            "pred_problem_type": pred_problem_type,
            "actual_problem_type": actual_problem_type,
            "problem_type_match": problem_type_match,

            "pred_difficulty": pred_difficulty,
            "actual_difficulty": actual_difficulty,
            "difficulty_match": difficulty_match,

            "pred_required_concepts": pred_required_concepts,
            "actual_required_concepts": actual_required_concepts,
            "concepts_match": concepts_match,

            "pred_solution_strategy": pred_solution_strategy,
            "actual_solution_strategy": actual_solution_strategy,
            "strategy_match": strategy_match,

            "pred_key_equations": pred_key_equations,
            "actual_key_equations": actual_key_equations,
            "equations_match": equations_match,

            "pred_final_answer": pred_final_answer,
            "actual_final_answer": actual_final_answer,
            "final_answer_match": final_answer_match,

            "pred_common_mistakes": pred_common_mistakes,
            "actual_common_mistakes": actual_common_mistakes,
            "mistakes_match": mistakes_match,

            "overall_match": overall_match,

            "prediction_raw": pred,
            "reference_raw": reference
        })

    return pd.DataFrame(rows)
