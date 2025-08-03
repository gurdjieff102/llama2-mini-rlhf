from transformers import AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch
import os

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 环境设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# 生成模型和分词器
causal_model_name = "gpt2"
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(causal_model_name).to(device)
ppo_tokenizer = AutoTokenizer.from_pretrained(causal_model_name)
ppo_tokenizer.pad_token = ppo_tokenizer.eos_token  # 设置pad_token

# 奖励模型
reward_model_dir = "./reward_model"
reward_pipe = pipeline("sentiment-analysis", 
                      model=reward_model_dir, 
                      tokenizer=reward_model_dir,
                      device=0 if torch.cuda.is_available() else -1)

# PPO配置
ppo_config = PPOConfig(
    batch_size=1,               # 匹配单样本处理
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
)

# 创建PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=ppo_model,
    ref_model=None,
    tokenizer=ppo_tokenizer
)

# 测试prompts
prompts = [
    "Tell me a joke about AI.",
    "What is the purpose of life?",
]

epochs = 4
for epoch in range(epochs):
    for prompt in prompts:
        # 编码prompt
        inputs = ppo_tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        # 生成响应
        ppo_model.eval()
        with torch.no_grad():
            response_ids = ppo_model.generate(
                input_ids, 
                max_new_tokens=20, 
                pad_token_id=ppo_tokenizer.eos_token_id,
                attention_mask=attention_mask
            )
        ppo_model.train()
        
        # 解码为字符串用于奖励模型
        response_text = ppo_tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        # 奖励模型评分
        reward_score = reward_pipe(response_text)[0]["score"]
        
        # 准备输入：query和response都必须是tensor
        input_tensors = [input_ids.squeeze(0)]  # 形状 [seq_len]
        response_tensors = [response_ids.squeeze(0)]  # 形状 [response_seq_len]
        
        # 修复：将奖励分数转换为tensor
        rewards = [torch.tensor(reward_score, device=device)]
        
        # PPO训练step
        ppo_trainer.step(input_tensors, response_tensors, rewards)
        
        print(f"Prompt: {prompt}\nResponse: {response_text}\nReward: {reward_score:.4f}\n")

# 保存模型
ppo_trainer.model.save_pretrained("./ppo_model")
print("✅ PPO微调完成，模型已保存至 ./ppo_model")