from transformers import AlbertForSequenceClassification, AlbertTokenizer
from datasets import load_dataset
import torch

# 加载微调后的模型
model_path = "./albert_imdb_model"  # 替换为你保存模型的路径
model = AlbertForSequenceClassification.from_pretrained(model_path)
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

# 加载IMDb测试集
dataset = load_dataset("./imdb", split="test")

# 定义函数进行推断
def predict_sentiment(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# 在整个测试集上进行推断并实时显示准确率
correct_predictions = 0
total_examples = len(dataset)

for idx, example in enumerate(dataset):
    text = example["text"]
    label = example["label"]
    predicted_class = predict_sentiment(text)

    if predicted_class == label:
        correct_predictions += 1

    if idx % 50 == 0:
        accuracy = correct_predictions / (idx + 1)
        print(f"\rProgress: {idx}/{5000}, Accuracy: {accuracy:.2%}", end="")

    if idx >= 5000:
        break

print("\nFinal Accuracy on IMDb test set:", correct_predictions / 5000)
