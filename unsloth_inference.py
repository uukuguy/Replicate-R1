#!/usr/bin/env python
from typing import Union, List
from unsloth import FastLanguageModel, PatchFastRL


def model_inference(
    prompt_or_messages: Union[str, List[dict]],
    model,
    tokenizer,
    lora_request=None,
    gen_kwargs: dict = None,
    system_prompt: str = None
):
    if isinstance(prompt_or_messages, str):
        prompt = prompt_or_messages
        messages = [{
            "role": "system",
            "content": system_prompt
        }] if system_prompt is not None else []
        messages.extend([{
            "role": "user",
            "content": prompt
        }])

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=gen_kwargs.get("temprature", 0.8),
        top_p=gen_kwargs.get("top_p", 0.95),
        max_tokens=gen_kwargs.get("max_tokens", 1024),
    )
    response = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=model.load_lora(lora_request) if lora_request else None,
    )

    generated_text = response[0].outputs[0].text
    return generated_text
