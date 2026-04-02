import time
import torch


def generate_response(
    model,
    tokenizer,
    user_prompt,
    max_new_tokens=180,
    do_sample=False,
    benchmark=False,
):
    messages = [{"role": "user", "content": user_prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    start = time.perf_counter() if benchmark else None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    end = time.perf_counter() if benchmark else None

    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if not benchmark:
        return generated_text

    total_time_s = end - start
    generated_tokens = generated_ids.shape[0]
    tokens_per_second = generated_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "text": generated_text,
        "total_time_s": total_time_s,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
    }

def format_example(example, tokenizer):
    text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    return {"text": text}