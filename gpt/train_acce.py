#encoding: utf-8
import torch
from accelerate import Accelerator
from model import GPTLMModel, GPTLMLoss, get_dataloader, get_model_size, get_tflops
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast
# 设置训练参数
batch_size = 3
learning_rate = 5e-4
num_epochs = 10


# 创建模型和损失函数
model = GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, max_seq_len=1024)
loss_fn = GPTLMLoss()
# 初始化Accelerator
accelerator = Accelerator()


optimizer = optim.SGD(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = accelerator.prepare_scheduler(scheduler)
optimizer = accelerator.prepare_optimizer(optimizer)

model = accelerator.prepare_model(model)

# 创建数据加载器
dataloader = get_dataloader(vocab_size=50257, seq_length=1024, batch_size=batch_size, data_size=256, num_workers=4)
dataloader = accelerator.prepare_data_loader(dataloader)

gradient_accumulation_steps = 2  # 每累积4个批次执行一次优化步骤
accumulated_loss = 0.0
# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    scheduler.step()
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        input_ids = batch[0]
        attention_mask = torch.ones_like(input_ids)
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, input_ids)  # 计算损失
        
        loss = loss / gradient_accumulation_steps
        accumulated_loss += loss

        if (batch) % gradient_accumulation_steps == 0 or batch == len(dataloader) - 1:
            optimizer.zero_grad()
            # 执行反向传播和优化步骤
            accelerator.backward(accumulated_loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 可选的梯度裁剪
            optimizer.step()
            accumulated_loss = 0.0
        
        total_loss += loss.item()
        optimizer.zero_grad()
        
        # 执行反向传播和优化步骤
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")

# 估计模型的大小和计算量
model_size = get_model_size(model)
tflops = get_tflops(model_size, batch_size, 1024, 1.0)  # 假设每步耗时1秒
print(f"Estimated Model Size: {model_size} parameters")
print(f"Estimated TFLOPs: {tflops:.2f}")
