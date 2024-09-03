from llama_cpp import Llama


MODEL_PATH = "/Users/xiaoyan.zang/llama-2-7b.Q5_K_M.gguf"

model: Llama = Llama(
    model_path=MODEL_PATH,
    verbose=False,
)



token_count: int = 0
for result in model.create_completion(
    prompt="What is the sound of one hand clapping?",
    max_tokens=200,
    stream=True,
    stop=[".", "?", "!"],
):
    if token_count % 50 == 0:
        print("")
    token_count = token_count + 1
    print(result["choices"][0]["text"], end="")



print("")


# Let's define the function we are going to use as our
# stopping criteria. At the moment I'm just going to
# return false, and set a breakpoint on this line
def should_stop(input_ids, logits) -> bool:
    return token_count == 100


# Now we'll do the rest as we did previously, this
# time setting up our stopping_criteria callback. I want
# to use a temperature of zero just for a demonstration
token_count: int = 0
for result in model.create_completion(
    prompt="What is the sound of one hand clapping?",
    max_tokens=200,
    stream=True, # 启用流式响应。这意味着模型会逐步生成并输出 token，而不是一次性返回整个响应。
    temperature=0,
    stopping_criteria=should_stop,
):
    if token_count % 50 == 0:
        print("")
    token_count = token_count + 1
    print(result["choices"][0]["text"], end="")