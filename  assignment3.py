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


from dataclasses import dataclass


@dataclass
class MailingAddress:
    name: str  # Full name, e.g. Dr. Christopher Brooks
    business_name: (
        str | None
    )  # Optional business name, e.g. School of Information, University of Michigan
    street_number: int  # Numeric address value, e.g. 105
    street_text: str  # Street information other than numeric address, e.g. S. State St.
    city: str  # City name, e.g. Ann Arbor
    state: str  # State name, only two letters, e.g. MI for Michigan
    zip_code_short: str  # The first five digits of the zip code, e.g. 48109, as a string value, since it could start with 0
    zip_code_long: (
        str | None
    )  # The extended zip code (optional) which is the full zip code, e.g. 48109-1285


prompt='''
Dr. Christopher Brooks
    School of Information, University of Michigan
    105 S. State St.
    Ann Arbor, MI
    48109-1285


Dr. Emily 
    Johnson Department of Computer Science, Stanford University
    353 Serra Mall
    Stanford, CA
    94305-2121


Dr. Michael Lee
    Department of Electrical Engineering, Massachusetts Institute of Technology
    77 Massachusetts Ave
    Cambridge, MA
    02139-4307

'''

grammar= r'''
root ::= (name businessname streetnum streettext city state zipshort ziplong "\n"){5} 
        # A sentence is just alphanumerica latin values, plus punctuation and whitespace
        # No parentheticals are allowed in a sentence, but a comma and hyphen are
        # A question should be a sentence or two, no more.
name ::= "Dr. " [A-Z][a-z]+ " " [A-Z][a-z]+ "\n"
businessname ::= [A-Z][a-z, -]* ", " [A-Z][a-z, -]* "\n"
streetnum ::= [0-9]+ " " 
streettext ::= [A-Z][a-z]+ " " [A-Z][a-z]+ "\n"
city ::= [A-Za-z]+ ", "
state ::= [A-Z]{2} "\n"
zipshort ::= [0-9]{5} "-"
ziplong ::= [0-9]{4} "\n"
'''

for item in model.create_completion(prompt,
    grammar=LlamaGrammar.from_string(grammar=grammar), 
    stream=True, 
    max_tokens=4098):
    print(item["choices"][0]["text"], end="")