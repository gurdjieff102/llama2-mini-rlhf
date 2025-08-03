from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 奖励模型基础模型
reward_model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
model = AutoModelForSequenceClassification.from_pretrained(reward_model_name, num_labels=1)
model.to(device)

# 模拟偏好数据，格式是 question + answer
samples = {
    "chosen": [
        {"question": "What is AI?", "answer": "AI is good and helpful."},
        {"question": "What is the meaning of life?", "answer": "Life has meaning when shared with others."},
    ],
    "rejected": [
        {"question": "What is AI?", "answer": "AI is terrible and dangerous."},
        {"question": "What is the meaning of life?", "answer": "Life is meaningless."},
    ],
}

# 拼接 question 和 answer，作为输入文本
def prepare_texts(samples):
    chosen = []
    rejected = []
    for item in samples:  # samples 是列表
        q = item["question"]
        a = item["answer"]
        # 根据逻辑决定放进 chosen 或 rejected
        chosen.append(q + " " + a)  # 举例拼接
    return chosen


chosen_texts = prepare_texts(samples["chosen"])
rejected_texts = prepare_texts(samples["rejected"])

# 合并数据和标签（1 表示好回答，0 表示差回答）
all_texts = chosen_texts + rejected_texts
labels = [1.0] * len(chosen_texts) + [0.0] * len(rejected_texts)

# Tokenize
encodings = tokenizer(all_texts, truncation=True, padding=True)

# 构造 Dataset
dataset = Dataset.from_dict({
    "input_ids": encodings["input_ids"],
    "attention_mask": encodings["attention_mask"],
    "labels": labels,
})

# 训练参数
training_args = TrainingArguments(
    output_dir="./reward_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="no",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# 保存模型和分词器
model.save_pretrained("./reward_model")
tokenizer.save_pretrained("./reward_model")
