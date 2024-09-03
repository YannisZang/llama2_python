#Generate for me a list of 5 things to do in your hometown (or mine if you prefer, Ann Arbor Michigan!).

import re
from llama_cpp import Llama
from llama_cpp.llama_types import *
from llama_cpp.llama_grammar import *




def generate_trip_recommendations() -> list[str]:
    results: list[str] = [""] * 5

    MODEL_PATH = "/Users/xiaoyan.zang/llama-2-7b.Q5_K_M.gguf"

    model: Llama = Llama(
        model_path=MODEL_PATH,
        verbose=False,
        n_ctx=2048
    )
    # Our prompt will just be a list of Mushroom questions in Aiken format
    prompt='''
    1. Visit Central Park. Take a leisurely walk through the iconic paths. Enjoy a picnic by the serene lakes.
    2. Explore the Metropolitan Museum of Art. Discover art pieces from ancient to modern times. Marvel at the impressive architecture of the museum.
    3. See a Broadway Show. Experience the excitement of live theater. Choose from popular musicals and plays.
    
    '''

    grammar= r'''
    root ::= (index things "\n"){5}
    # A sentence is just alphanumerica latin values, plus punctuation and whitespace
    # No parentheticals are allowed in a sentence, but a comma and hyphen are
    sentence ::= [A-Z] [A-Za-z0-9 ,-]* (".")
    # A question should be a sentence or two, no more.
    index ::= [0-9] "."
    things ::= sentence " " sentence " " sentence
    '''
    index: int = 0

    for item in model.create_completion(prompt,
        grammar=LlamaGrammar.from_string(grammar=grammar), 
        stream=True, 
        max_tokens=4096):
        print(item["choices"][0]["text"], end="")
        if item["choices"][0]["text"] != "\n":
            results[index] = results[index] + item["choices"][0]["text"]
            continue
        elif index < 4:
            index += 1
    
        
        
        # YOUR CODE HERE
    return results

result = generate_trip_recommendations()
print(result)