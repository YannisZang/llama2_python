from llama_cpp import Llama
from llama_cpp.llama_types import *

MODEL_PATH = "/Users/xiaoyan.zang/llama-2-7b.Q4_0.gguf"

model: Llama = Llama(
    model_path=MODEL_PATH,
    verbose=False,
    logits_all=True
)

# As the temperature was set to 0 the very first token, the one which minimized the absolute value of the logprob, was chosen. We can experiment
# a bit with this -- let's turn up the heat and see what happens
result: Completion = model.create_completion(
    prompt="The capital of Michigan is ", logprobs=3, max_tokens=16, temperature=0.7
)

details: CompletionLogprobs = result["choices"][0]["logprobs"]

# Here we can look at the list of tokens, their probabilities, and their offset
# in the prompt+response
print(details["tokens"])
print()  # little whitespace to see better

# iterate through results
for token_choice in details["top_logprobs"]:
    print(token_choice)

# So here we can see that the sequence returned isn't made up of each of the lowest tokens in the candidate sets. 
# In fact, depending on a little bit of randomness, you might even see that several of the candidate sets have more than the
# 3 tokens in them -- that the high temperature resulted in the model choosing a token that wasn't in the top three
# options. This important to note too -- even though we set the logprobs to three, that's not the complete candidate set of tokens, but just the ones which are being returned to us programmatically.


# top_k 
# top_k 是在自然语言处理和机器学习中常用的一个参数，特别是在生成文本的任务中，它用来控制生成的随机性。
# top_k 参数定义了在生成下一个 token（文本片段或单词）时，从模型预测的概率分布中选择概率最高的 k 个 token，
# 然后从这 k 个 token 中随机选择一个作为输出。这种方法可以防止生成文本时出现不常见或不合理的单词，同时保持一定的多样性。

# temperature: temperature 控制选择高概率或低概率 token 的倾向，影响生成文本的随机性。temperature 越高，生成的文本越随机；
# temperature 越低，生成的文本越确定。top_k 可以在设定的范围内限制这种随机性，即使 temperature 较高，top_k 也确保只在高概率选项中选择。


# Let's now constrain the set of choices the model can consider by setting top_k
# to two. In this world the model only has a choice between the two best fitting
# tokens for a position in the output sequence.
result: Completion = model.create_completion(
    prompt="The capital of Michigan is ",
    logprobs=3,
    max_tokens=16,
    temperature=1.0,
    top_k=2,
)

details: CompletionLogprobs = result["choices"][0]["logprobs"]

# Here we can look at the list of tokens, their probabilities, and their offset
# in the prompt+response
print(details["tokens"])
print()  # little whitespace to see better

# iterate through results
for token_choice in details["top_logprobs"]:
    print(token_choice)


# top_p
# top_p 是文本生成任务中常用的一个参数，用来控制生成文本的多样性和质量。top_p 也被称为“核采样”（nucleus sampling），它是一种基于概率的采样方法，用来选择候选 token 的子集。通过设定 top_p 参数，可以动态地限制模型在生成下一个 token 时考虑的候选 token 数量，使得生成的文本在合理性和多样性之间取得平衡。

# 1. top_p 的基本概念
# 作用: top_p 控制生成下一个 token 时，只考虑使得累积概率超过某个阈值的最小 token 集合。这个阈值由 top_p 的值决定。

# 示例: 假设模型在给定上下文后预测下一个 token 的概率分布是：

# Token	Probability
# "apple"	0.4
# "banana"	0.3
# "cherry"	0.1
# "date"	0.08
# "elder"	0.07
# "fig"	0.05
# 如果设置 top_p=0.9，模型将从概率累计达到 0.9 的最小 token 集合中选择，即 ["apple", "banana", "cherry"]，
# 因为它们的累积概率为 0.4 + 0.3 + 0.1 = 0.8。然后从这个集合中随机选择一个 token 作为输出。

result: Completion = model.create_completion(
    prompt="The capital of Michigan is ",
    logprobs=3,
    max_tokens=16,
    temperature=0.7,
    top_p=0.5,
)

details: CompletionLogprobs = result["choices"][0]["logprobs"]

# Here we can look at the list of tokens, their probabilities, and their offset
# in the prompt+response
print(details["tokens"])
print()

# iterate through results
for token_choice in details["top_logprobs"]:
    print(token_choice)