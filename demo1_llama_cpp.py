from llama_cpp import Llama
from llama_cpp.llama_types import *

MODEL_PATH = "/Users/xiaoyan.zang/llama-2-7b.Q4_0.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    verbose = False
)


result: Completion = llm.create_completion(prompt="The capital of Michigan is ")

# The Completion type has a choices key which shows us the list of
# responses the LLM generated, let's take a look
print(result["choices"])

# Let's try a few different temperature values. 
# Temperature is a parameter from zero to one where values closer to 0 cause the model
# will behave more deterministically and values closer to one cause the model to behave more non-deterministic, and creative.
temps: list[float] = [0.0, 0.5, 1.0]

# Now, for each of these temperatures, let's do three completions
prompt: str = "The planets in the solar system include "
for temp in temps:
    for i in range(0, 3):
        result: Completion = llm.create_completion(prompt=prompt, temperature=temp)
        print(f'temp={temp}, run={i}, result: {result["choices"][0]["text"]}')



# leave the temperature at it's default value, which is 0.8
# by default, token return 16, change token to -1 to generate as many tokens as available
result: Completion = llm.create_completion(prompt=prompt, max_tokens=20)
print(result["choices"][0]["text"])


# I'm going to create a new model with a nice large context size
model: Llama = Llama(model_path=MODEL_PATH, verbose=False, n_ctx=4096)


# If we pass the stream=True parameter to create_completion() we will get back
# an iterator of CreatCompletionStreamResponse objects, which are just typed
# dictionaries similar to the Completion type
token_count: int = 0
for result in model.create_completion(
    prompt="Some fun things to do for vacation in the state of Michigan includes ",
    max_tokens=-1,
    stream=True,
):
    # I'm only going to print a newline every 50 tokens or so
    if token_count % 50 == 0:
        print("")
    token_count = token_count + 1
    print(result["choices"][0]["text"], end="")





