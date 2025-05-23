# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# # Keep chat history in tokenized form
# chat_history_ids = None
# step = 0

# print("Type 'quit' to exit.\n")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         break

#     # Encode the user input + add end-of-string token
#     new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

#     # Append to chat history
#     bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if step > 0 else new_input_ids

#     # Prepare attention mask
#     attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

#     # Generate a response with attention mask
#     chat_history_ids = model.generate(
#         bot_input_ids,
#         attention_mask=attention_mask,
#         max_length=1500,
#         min_length=10,
#         pad_token_id=tokenizer.eos_token_id,
#         do_sample=True,
#         top_k=40,
#         top_p=0.85,
#         temperature=0.7,
#         repetition_penalty=1.2
#     )


#     # Decode and print the response
#     response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
#     print(f"Bot: {response}")
#     step += 1

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         break

#     prompt = f"{user_input}"
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids

#     output_ids = model.generate(input_ids, max_new_tokens=200)
#     response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     print("Bot:", response)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "EleutherAI/gpt-j-6B"  # <-- stronger than DialoGPT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,      # use fp16 if your GPU supports it
    device_map="auto"               # auto–place layers on GPU/CPU
)

chat_history_ids = None
print("Type 'quit' to exit.\n")

while True:
    prompt = input("You: ")
    if prompt.lower() in ("quit","exit"):
        break

    new_ids = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt").input_ids.to(model.device)
    bot_input = torch.cat([chat_history_ids, new_ids], dim=-1) if chat_history_ids is not None else new_ids

    chat_history_ids = model.generate(
        bot_input,
        max_length=1024,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    reply = tokenizer.decode(chat_history_ids[:, bot_input.shape[-1]:][0], skip_special_tokens=True)
    print("Bot:", reply)
