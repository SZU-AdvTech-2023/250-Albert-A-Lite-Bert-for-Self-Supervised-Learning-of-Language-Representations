# from transformers import pipeline
# unmasker = pipeline('fill-mask', model='albert-base-v1')
# print(unmasker("Hello I'm a [MASK] model."))



from transformers import AlbertTokenizer, AlbertModel
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
model = AlbertModel.from_pretrained("albert-base-v1")
text = "I hate you."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)