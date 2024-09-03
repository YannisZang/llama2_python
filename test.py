import re
from llama_cpp import Llama
from llama_cpp.llama_types import *
from llama_cpp.llama_grammar import *


MODEL_PATH = "/Users/xiaoyan.zang/llama-2-7b.Q5_K_M.gguf"

model: Llama = Llama(
    model_path=MODEL_PATH,
    verbose=False,
    n_ctx=2048
)

results: list[str] = [""] * 10
# Our prompt will just be a list of Mushroom questions in Aiken format
prompt='''Chen Baiqing
Wang Yilin
Song Ke
Ren Pengfei
Fang Xingyong
Wu Yu

'''

grammar= r'''
root ::= (name "\n"){5}
# A sentence is just alphanumerica latin values, plus punctuation and whitespace
# No parentheticals are allowed in a sentence, but a comma and hyphen are
sentence ::= [A-Z][A-Za-z]* 
# A question should be a sentence or two, no more.
name ::= sentence " " sentence
# An answer is a capital letter followed by a close parens and then a sentence
'''
index: int = 0

for item in model.create_completion(prompt,
    grammar=LlamaGrammar.from_string(grammar=grammar), 
    stream=True, 
    max_tokens=515):
    print(item["choices"][0]["text"], end="")
    
    
