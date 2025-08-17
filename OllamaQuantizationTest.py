# Courtesy: ChatGPT

import time
from ollama import Client

# Connect to local Ollama instance
client = Client(host='http://localhost:11434')

# Test prompt
prompt = "Explain in two sentences why Intel Arc GPUs with IPEX are useful for AI inference."

# Function to benchmark model generation
def benchmark(model_name: str, max_tokens: int = 50):
    print(f"\n--- Testing model: {model_name} ---")
    start = time.time()
    response = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens}
    )
    end = time.time()
    output = response['message']['content']
    latency = end - start
    print(f"Output: {output.strip()}")
    print(f"Tokens generated: {max_tokens}")
    print(f"Time taken: {latency:.2f} sec")
    print(f"Throughput: {max_tokens/latency:.2f} tokens/sec")

# Run benchmarks on different quantized builds (if available locally)
models_to_test = [
    "gpt-oss:20b",           # default
    "deepseek-r1:8b",    # 4-bit quantized
    "llama3.2:latest",      # 5-bit quantized
#    "gpt-oss-20b:q8_0",      # 8-bit quantized
]

for m in models_to_test:
    try:
        benchmark(m)
    except Exception as e:
        print(f"Skipping {m} (not installed): {e}")

