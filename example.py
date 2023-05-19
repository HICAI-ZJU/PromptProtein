import torch
from models import openprotein_promptprotein
from utils import PromptConverter

model, dictionary = openprotein_promptprotein('/path/to/PromptProtein.pt')
model = model.eval().cuda()
converter = PromptConverter(dictionary)


data = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
]

prompts = ['<seq>']

encoded_sequences = converter(data, prompt_toks=prompts)

with torch.no_grad():
    logits = model(encoded_sequences.cuda(), with_prompt_num=1)['logits']

