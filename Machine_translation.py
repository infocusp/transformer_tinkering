from transformers import XLMWithLMHeadModel, XLMTokenizer

# Define the model and tokenizer
model_name = 'xlm-mlm-en-2048'
tokenizer = XLMTokenizer.from_pretrained(model_name)
model = XLMWithLMHeadModel.from_pretrained(model_name)

def generate_text(prompt):
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text using the model
    outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)

    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# Example prompts
prompts = [
    "Artificial intelligence is",
    "In the future, cars will",
    "Climate change affects",
]

for prompt in prompts:
    print(f"Prompt: {prompt}")
    print(f"Generated Text: {generate_text(prompt)}")
    print()
