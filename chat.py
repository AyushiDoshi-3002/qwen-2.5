"""
Simple chat CLI for Qwen2.5-3B-Instruct using Hugging Face transformers.

Usage:
    pip install -r requirements.txt
    python chat.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def main():
    device, dtype = get_device_and_dtype()
    print(f"Loading {MODEL_NAME} on {device} ({dtype})...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device,
    )

    system_prompt = "You are a helpful assistant."
    history = [{"role": "system", "content": system_prompt}]

    print("Model loaded. Type 'exit' to quit, 'reset' to clear history.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            history = [{"role": "system", "content": system_prompt}]
            print("History cleared.\n")
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        print("Assistant: ", end="", flush=True)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            streamer=streamer,
        )
        print()

        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
