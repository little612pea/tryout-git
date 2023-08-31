#encoding:utf-8
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from model import GPTLMModel, GPTLMLoss, get_dataloader, get_model_size, get_tflops

# 设置训练参数
batch_size = 2
learning_rate = 5e-4
num_epochs = 10

# 初始化进程组
dist.init_process_group(backend='nccl', init_method='env://')
# init_method='env://'表示使用环境变量来初始化连接信息
# 例如：os.environ['MASTER_ADDR'] = '
# 

# 获取当前进程的rank和进程总数
world_size = dist.get_world_size()
rank = dist.get_rank()

# 创建模型和损失函数
model = GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, max_seq_len=1024)
loss_fn = GPTLMLoss()

# 将模型放置在多个GPU上
device_ids = list(range(torch.cuda.device_count()))
device = torch.device(f"cuda:{rank}")
model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

# 创建数据加载器
dataloader = get_dataloader(vocab_size=50257, seq_length=1024, batch_size=batch_size * world_size, data_size=256, num_workers=4)

# 选择优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        # 将数据和标签移动到GPU上
        input_ids = batch[0].to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        # 计算分块大小
        bucket_size = input_ids.size(0) // world_size
        
        # 分块
        input_ids = input_ids[rank * bucket_size : (rank + 1) * bucket_size]
        attention_mask = attention_mask[rank * bucket_size : (rank + 1) * bucket_size]
        
        # 计算logits
        logits = model(input_ids, attention_mask)
        
        # 计算损失
        loss = loss_fn(logits, input_ids)
        loss = loss.mean()  # 对所有GPU上的损失求平均
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录总损失
        total_loss += loss.item()
    
    # 在所有进程上对总损失求和
    total_loss = torch.tensor(total_loss).to(device)
    dist.all_reduce(total_loss)
    if rank == 0:
        average_loss = total_loss.item() / len(dataloader) / world_size
        # 打印平均损失
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")

# 估计模型的大小和计算量
model = model.module.to(device)
model_size = get_model_size(model)
tflops = get_tflops(model_size, batch_size, 1024, 1.0)  # 假设每步耗时1秒

# 在rank为0的进程上打印模型大小和计算量
if rank == 0:
    print(f"Estimated Model Size: {model_size} parameters")
    print(f"Estimated TFLOPs: {tflops:.2f}")
