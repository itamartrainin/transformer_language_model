from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

fill_mask = pipeline(
    "fill-mask",
    model="bert-base-uncased",
    tokenizer="bert-base-uncased"
)

result = fill_mask("Today is a sunny day and the weather is [MASK].")

print(result)


#%%
import torch
from transformers import AutoTokenizer, BertForMaskedLM

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = BertForMaskedLM.from_pretrained("bert-base-uncased")

input_idx = tok.encode(f"The {tok.mask_token} were the best rock band ever.")
logits = bert(torch.tensor([input_idx]))[0]
prediction = logits[0].argmax(dim=1)
print(tok.ids_to_tokens[prediction[2].numpy().tolist()])