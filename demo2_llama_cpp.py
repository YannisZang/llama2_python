# parameterizing the Llaama 2 Model

from llama_cpp import Llama
from llama_cpp.llama_types import *

MODEL_PATH = "/Users/xiaoyan.zang/llama-2-7b.Q4_0.gguf"


# 设置 verbose=False 意味着在执行模型相关操作时，不需要打印详细的日志或信息。这通常用于减少终端输出，使程序输出更加简洁。
# logits_all 是另一个布尔参数。logits 通常指的是在模型的输出层中，未经过激活函数的原始输出值（通常是神经网络中的原始预测分数）。
# 设置 logits_all=True 可能表示模型在预测时返回所有的 logits 而不是某些特定的输出。这在需要详细分析模型行为或者计算特定指标时可能很有用。
model: Llama = Llama(
    model_path=MODEL_PATH,
    verbose=False,
    logits_all=True
)

# When we send a prompt to the model we can pass in the logprobs parameter which will determine how many alternative choices we will see. I'll just look at the top two choices. 
# And we'll set the temperature to 0, which means that we'll have the model always choose from that top 3 the very best choice

result: Completion = model.create_completion(prompt="The capital of Michigan is ", logprobs=3, max_tokens=16, temperature=0)


# Take out the response content from the Completion object
item = result["choices"][0]

# Print out the response text
print(item["text"])

# The individual token probabilities are in the CompletionLogprobs object
details: CompletionLogprobs = item["logprobs"]


# Here we can look at the list of tokens, their probabilities, and their offset
# in the prompt+response
print(details["tokens"])  # these will actually be encoded (text) copies of tokens!
print(
    details["token_logprobs"]
)  # the probability of the token in the set of candidates
print(details["text_offset"])  # the location in the prompt+response


# We can actually see for each token what the alternatives were and what their
# logprob was. We can iterate through the set of tokens and, since I set logprobs=3,
# we will see the top three options for each token
for token_choice in details["top_logprobs"]:
    print(token_choice)