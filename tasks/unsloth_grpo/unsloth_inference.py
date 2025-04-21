#!/usr/bin/env python
from typing import Union, List
from unsloth import FastLanguageModel, PatchFastRL
import torch
from loguru import logger


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
    if gen_kwargs is not None:
        sampling_params = SamplingParams(
            temperature=gen_kwargs.get("temprature", 0.8),
            top_p=gen_kwargs.get("top_p", 0.95),
            max_tokens=gen_kwargs.get("max_tokens", 1024),
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
        )
    response = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=model.load_lora(lora_request) if lora_request else None,
    )

    generated_text = response[0].outputs[0].text
    return generated_text

def unsloth_inference(prompt_or_messages, model_path, lora_model_path, max_seq_length=2048):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_model_path, #model_path, 
        max_seq_length = max_seq_length,
        load_in_4bit = True,
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = 64,
        gpu_memory_utilization = 0.85, # Reduce if out of memory
    )
    model_weights_dir = "/opt/local/llm_models/huggingface.co/speechlessai/function_calling_qwen_7b_instruct_unsloth"
    logger.info(f"Saving model to {model_weights_dir}")
    model.save_pretrained_merged(model_weights_dir, tokenizer, save_method = "merged_16bit")
    logger.info(f"Model saved to {model_weights_dir}")
    # FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    # # text_streamer = TextStreamer(tokenizer)
    # # _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 64) 
    # generated_text = model_inference(prompt_or_messages=prompt_or_messages, model=model, tokenizer=tokenizer, lora_request=None)
    # return generated_text

def main():
    model_path = "/opt/local/llm_models/huggingface.co/speechlessai/function_calling_qwen_7b_instruct"
    lora_model_path = "outputs_grpo/checkpoint-2000"
    prompt = "Which is bigger, 9.9 or 9.11"
    generated_text = unsloth_inference(prompt, model_path=model_path, lora_model_path=lora_model_path)
    print(generated_text)

if __name__ == "__main__":
    main()

"""
    --enable-lora \
    --lora-modules tools-sft=$HOME/sandbox/LLM/speechless.ai/speechless/tasks/synthesize_tools_sft/function_calling_qwen_7b_instruct/outputs_grpo/checkpoint-2000
"""