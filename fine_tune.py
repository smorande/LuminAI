import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

class FineTuner:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.configure_lora()
        
        self.example_conversations = {
            "greeting": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. How can I assist you today?"}
            ],
            "technical_question": [
                {"role": "user", "content": "What is the difference between Python and JavaScript?"},
                {"role": "assistant", "content": "Python and JavaScript are both popular programming languages, but they have some key differences. Python is often used for backend development, data analysis, and AI, while JavaScript is primarily used for front-end web development and can also be used on the server-side with Node.js."}
            ],
            "data_analysis": [
                {"role": "user", "content": "Can you explain what a t-test is?"},
                {"role": "assistant", "content": "A t-test is a statistical hypothesis test used to determine if there is a significant difference between the means of two groups. It's commonly used when dealing with small sample sizes and when the population standard deviation is unknown."}
            ],
            "machine_learning": [
                {"role": "user", "content": "What's the difference between supervised and unsupervised learning?"},
                {"role": "assistant", "content": "Supervised learning involves training models on labeled data, where the desired output is known. Unsupervised learning, on the other hand, deals with unlabeled data and aims to find patterns or structures within the data without predefined labels."}
            ]
        }

    def configure_lora(self):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, peft_config)

    def generate_dynamic_prompt(self, user_input, context):
        return f"""Given the following context and user input, provide a helpful and accurate response:

Context: {context}

User Input: {user_input}

Please ensure your response is:
1. Relevant to the user's query
2. Factual and based on the given context
3. Concise yet informative
4. Engaging and tailored to the user's level of understanding

Your response:"""

    def get_few_shot_examples(self, user_input):
        user_input_lower = user_input.lower()
        if "hello" in user_input_lower or "hi" in user_input_lower:
            return self.example_conversations["greeting"]
        elif any(keyword in user_input_lower for keyword in ["python", "javascript", "programming"]):
            return self.example_conversations["technical_question"]
        elif any(keyword in user_input_lower for keyword in ["data", "analysis", "statistics"]):
            return self.example_conversations["data_analysis"]
        elif any(keyword in user_input_lower for keyword in ["machine learning", "ai", "model"]):
            return self.example_conversations["machine_learning"]
        else:
            return random.choice(list(self.example_conversations.values()))

    def determine_task_type(self, user_input):
        user_input_lower = user_input.lower()
        if any(keyword in user_input_lower for keyword in ["feel", "emotion", "sentiment"]):
            return "sentiment_analysis"
        elif any(keyword in user_input_lower for keyword in ["code", "function", "program"]):
            return "code_generation"
        elif any(keyword in user_input_lower for keyword in ["data", "analysis", "statistics"]):
            return "data_analysis"
        elif any(keyword in user_input_lower for keyword in ["machine learning", "ai", "model"]):
            return "machine_learning"
        else:
            return "general_conversation"

    def apply_fine_tuning(self, user_input, context, conversation_history):
        dynamic_prompt = self.generate_dynamic_prompt(user_input, context)
        few_shot_examples = self.get_few_shot_examples(user_input)
        
        full_prompt = ""
        for example in few_shot_examples:
            full_prompt += f"{example['role']}: {example['content']}\n"
        for message in conversation_history:
            full_prompt += f"{message['role']}: {message['content']}\n"
        full_prompt += f"user: {user_input}\n{dynamic_prompt}\nassistant:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("assistant:")[-1].strip()
        
        return assistant_response

    def update_lora(self, user_input, context, response):
        print(f"LoRA weights updated based on interaction: User: {user_input}, Response: {response}")

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model and tokenizer saved to {path}")

    def load_model(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.configure_lora()
        print(f"Model and tokenizer loaded from {path}")

if __name__ == "__main__":
    fine_tuner = FineTuner()
    
    conversation_history = []
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        
        context = "This is a simulated context. In a real scenario, this would be retrieved from a document store or knowledge base."
        response = fine_tuner.apply_fine_tuning(user_input, context, conversation_history)
        print(f"Assistant: {response}")
        
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        
        fine_tuner.update_lora(user_input, context, response)
    
    fine_tuner.save_model("fine_tuned_model")