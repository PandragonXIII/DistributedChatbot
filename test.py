import torch
print(torch.version.cuda)
import yaml
print(torch.cuda.is_available())
with open("./settings.yaml") as f:
    settings = yaml.safe_load(f)

from transformers import AutoModelForCausalLM, AutoTokenizer

qwen_path = settings["Qwen25_15_instruct_AWQ"]["path"]
# Load the model in half-precision on the available device(s)
model = AutoModelForCausalLM.from_pretrained(qwen_path, device_map="cuda", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(qwen_path)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)