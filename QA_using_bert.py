import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# Load the pretrained BERT model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def answer_question(question, context):
    # Tokenize the input
    inputs = tokenizer(question, context, return_tensors='pt')

    # Get the answer from the model
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Find the tokens with the highest `start` and `end` scores
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1 

    # Combine the tokens in the answer and return it
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    return answer

# Here are some example questions and a context paragraph:
context = """
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. It has become an essential part of the technology industry. 
Research associated with AI is highly technical and specialized. 
The core problems of AI include programming computers for certain traits such as knowledge, reasoning, problem-solving, perception, learning, planning, and the ability to manipulate and move objects. 
AI is used in many ways today. For example, it's used in self-driving cars, in voice assistants like Alexa and Siri, and in recommendation systems used by companies like Netflix and Amazon.
"""

questions = [
    "What is Artificial Intelligence?",
    "What is the aim of AI?",
    "What are the core problems of AI?",
    "In what examples is AI used today?",
    "Which companies use AI in their recommendation systems?",
]

for question in questions:
    print(f"Question: {question}")
    print(f"Answer: {answer_question(question, context)}")
    print()

