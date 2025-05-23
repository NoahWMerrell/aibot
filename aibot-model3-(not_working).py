from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "EleutherAI/gpt-j-6B"  # <-- stronger than DialoGPT
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload"  # creates a folder named "offload" in your current directory
)

chat_history_ids = None
print("Type 'quit' to exit.\n")

while True:
    prompt = input("You: ")
    if prompt.lower() in ("quit","exit"):
        break

    # Use the history and specify the attention mask
    inputs = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt", padding=True)
    new_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    bot_input = torch.cat([chat_history_ids, new_ids], dim=-1) if chat_history_ids is not None else new_ids

    # Options for tweaking model
    chat_history_ids = model.generate(
        bot_input,
        attention_mask=attention_mask,
        max_length=1024,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Print response
    reply = tokenizer.decode(chat_history_ids[:, bot_input.shape[-1]:][0], skip_special_tokens=True)
    print("Bot:", reply)