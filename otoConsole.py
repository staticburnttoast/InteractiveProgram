import json
from llama_cpp import Llama
from datetime import datetime

model_path="model/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
llm = Llama(model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=2048,
            n_batch=512)

def load_information(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_memories(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_memory(memory, file_path="memories.json"):
    with open(file_path, 'w') as file:
        json.dump(memory, file, indent=4)

def add_to_short_term(memory, conversation):
    memory_entry = {
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation
    }

    memory['short_term'].append(memory_entry)
    if len(memory['short_term']) > 20:
        memory['short_term'].pop(0)

    save_memory(memory)

def promote_to_long_term(memory, conversation, importance="high"):
    memory_entry = {
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation,
        "importance": importance
    }

    memory['long_term'].append(memory_entry)
    save_memory(memory)

    memory['long_term'].append(memory_entry)
    save_memory(memory)

def build_prompt(user_input, memory, information):
    information = load_information('information.json')
    short_term_context = "\n".join([entry["conversation"] for entry in memory["short_term"]])
    long_term_context = "\n".join(
        [entry["conversation"] for entry in memory["long_term"] if entry["importance"] == "high"])

    prompt = (f"{information['system_prompt']}\n"
              f"{information['characters']}\n"
              f"{long_term_context}\n"
              f"{short_term_context}\n"
              f"User: {user_input}\nMachi:")
    return prompt

def clear_memory(memory, memory_type):
    if memory_type == 'short_term':
        memory['short_term'] = []
        print("Short-term memory cleared.")
    elif memory_type == 'long_term':
        memory['long-term'] = []
        print("Long-term memory cleared.")
    else:
        print("Invalid memory type. Specify 'short_term' or 'long_term'.")

    save_memory(memory)
    
def Otogi():
  information = load_information('information.json')
  memory = load_memories('memories.json')
  
  while True:
      usr_input = input("User: ")

      if usr_input.lower() == 'clear short_term':
          clear_memory(memory, 'short_term')
          continue
      elif usr_input.lower() == 'clear long_term':
          clear_memory(memory, 'long-term')
          continue

      prompt = build_prompt(usr_input, memory, information)
      response = llm(prompt,
                      max_tokens=150,
                      temperature=0.7,
                      top_k=50,
                      top_p=0.9,
                      repeat_penalty=1.2,
                      stop=["User:", "Machi:"])
      otoResponse = response['choices'][0]['text'].strip()
      print("\nMachi: ", otoResponse, "\n")

      add_to_short_term(memory, f"User: {usr_input}")
      add_to_short_term(memory, f"Otogi: {otoResponse}")

      if "important" in usr_input:
          promote_to_long_term(memory, f"Noted: {usr_input}")

Otogi()