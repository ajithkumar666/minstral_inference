from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime
device = "cuda" # the device to load the model onto
# device = "cpu" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
while True:
      query = input("Enter your prompt: ")
      messages = [
         {"role": "user", "content": str(query)}
      ]
      encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
      generated_ids = model.generate(encodeds, max_new_tokens=300, do_sample=False)
      decoded = tokenizer.batch_decode(generated_ids)
      print("Query: ",query)
      print("Result")
      print(decoded[0])
