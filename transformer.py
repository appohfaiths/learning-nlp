from transformers import BertConfig, BertModel, BertTokenizer

config = BertConfig()
model = BertModel(config)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

result = tokenizer("Using a Transformer network is simple")

# print(config)
# print(model)

print(f"The result is : {result}")