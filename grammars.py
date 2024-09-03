from llama_cpp import Llama
from llama_cpp.llama_types import *
from llama_cpp.llama_grammar import *


MODEL_PATH = "/Users/xiaoyan.zang/llama-2-7b.Q5_K_M.gguf"

model: Llama = Llama(
    model_path=MODEL_PATH,
    verbose=False,
    n_ctx=2048
)

# prompt will just be a list of Mushroom questions in Aiken format
prompt='''Most common edible wild mushrooms are which of the following colors?
A) White, cream or ivory
B) Red
C) Orange or yellow
D) Green
ANSWER: A

What is a sporangium?
A) The term for mushroom gills that contain spores
B) The fruiting body of the mushroom
C) The sac-like structure on which spore are produced
D) A slender, thread like hyphae
ANSWER: C

What is a mycelium?
A) Sporangia covered with colorful scales or ridges
B) Undifferentiated mass of branching filaments
C) The stem of the mushroom that carries spores from the fruiting body to the soil where it will grow and reproduce.
D) The cap of a mushroom
ANSWER: D

'''

grammar= r'''
root ::= (question answer+ epilog "\n")+
# A sentence is just alphanumerica latin values, plus punctuation and whitespace
# No parentheticals are allowed in a sentence, but a comma and hyphen are
sentence ::= [A-Z] [A-Za-z0-9 ,-]* ("." | "!" | "?")
# A question should be a sentence or two, no more.
question ::= sentence "\n" | sentence " " sentence "\n"
# An answer is a capital letter followed by a close parens and then a sentence
answer ::= [A-Z] ") " sentence "\n"
# The question closes with an epilog telling us the question has finished
epilog ::= "ANSWER: " [A-Z] "\n"
'''

# Finally, invoke a completion
result = model.create_completion(prompt,
    grammar=LlamaGrammar.from_string(grammar=grammar), 
    stream=True, 
    max_tokens=400)
for item in result:
    print(item['choices'][0]['text'], end="")