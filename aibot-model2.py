from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"{user_input}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    output_ids = model.generate(input_ids, max_new_tokens=200)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("Bot:", response)