from llama_cpp import Llama
from llama_cpp.llama_types import *
from llama_cpp.llama_chat_format import format_llama2, ChatFormatterResponse

MODEL_PATH = "/Users/xiaoyan.zang/llama-2-7b.Q5_K_M.gguf"

model: Llama = Llama(
    model_path=MODEL_PATH,
    verbose=False,
    n_ctx=2048
)

prompt = "What is the fifth planet in the solar system?"

# Now let's watch the results. Remember you need to increase the
# max_tokens as well as the context window or llama.cpp will cut
# off the reply
for response in model.create_completion(prompt, max_tokens=20, stream=True):
    result = response["choices"][0]
    print(result["text"], end="")

print()

# few shot prompting
# Few-shot prompt 是一种有效的方法，可以帮助 llama-cpp 和其他语言模型更好地理解用户的期望和任务要求。
# 通过在 prompt 中添加几个示例，可以显著提升模型生成的准确性和相关性。这种方法在需要模型在特定任务中表现出色时特别有用。

prompt = """Python 3 lambda question in JSON:
{"question":"The lambda keyword in python is:","correct_answer":"For declaring anonymous functions","incorrect_answer":"For mathematical operations"}

Python 3 def question in JSON:
{"question":"What does the 'def' keyword do?","correct_answer":"Define a function","incorrect_answer":"Declare variables"}

Python 3 assert question in JSON: 
"""

for response in model.create_completion(prompt, max_tokens=40, stream=True):
    result = response["choices"][0]
    print(result["text"], end="")

print()


prompt = """
{"python_3_topics" = ["lambda", "def", "assert", "with", "import"],questions=[
{"question":"The lambda keyword in python is:", "correct_answer":"For declaring anonymous functions", "incorrect_answer":"For mathematical operations"},
{"question":"What does the 'def' keyword do?", "correct_answer":"Define a function", "incorrect_answer":"Declare variables"},
{
"""

for response in model.create_completion(prompt, max_tokens=48, stream=True):
    result = response["choices"][0]
    print(result["text"], end="")