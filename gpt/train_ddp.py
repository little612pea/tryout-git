#encoding:utf-8
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from model import GPTLMModel, GPTLMLoss, get_dataloader, get_model_size, get_tflops

# ����ѵ������
batch_size = 2
learning_rate = 5e-4
num_epochs = 10

# ��ʼ��������
dist.init_process_group(backend='nccl', init_method='env://')
# init_method='env://'��ʾʹ�û�����������ʼ��������Ϣ
# ���磺os.environ['MASTER_ADDR'] = '
# 

# ��ȡ��ǰ���̵�rank�ͽ�������
world_size = dist.get_world_size()
rank = dist.get_rank()

# ����ģ�ͺ���ʧ����
model = GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, max_seq_len=1024)
loss_fn = GPTLMLoss()

# ��ģ�ͷ����ڶ��GPU��
device_ids = list(range(torch.cuda.device_count()))
device = torch.device(f"cuda:{rank}")
model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

# �������ݼ�����
dataloader = get_dataloader(vocab_size=50257, seq_length=1024, batch_size=batch_size * world_size, data_size=256, num_workers=4)

# ѡ���Ż���
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ѵ��ѭ��
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        # �����ݺͱ�ǩ�ƶ���GPU��
        input_ids = batch[0].to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        # ����ֿ��С
        bucket_size = input_ids.size(0) // world_size
        
        # �ֿ�
        input_ids = input_ids[rank * bucket_size : (rank + 1) * bucket_size]
        attention_mask = attention_mask[rank * bucket_size : (rank + 1) * bucket_size]
        
        # ����logits
        logits = model(input_ids, attention_mask)
        
        # ������ʧ
        loss = loss_fn(logits, input_ids)
        loss = loss.mean()  # ������GPU�ϵ���ʧ��ƽ��
        
        # ���򴫲����Ż�
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ��¼����ʧ
        total_loss += loss.item()
    
    # �����н����϶�����ʧ���
    total_loss = torch.tensor(total_loss).to(device)
    dist.all_reduce(total_loss)
    if rank == 0:
        average_loss = total_loss.item() / len(dataloader) / world_size
        # ��ӡƽ����ʧ
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")

# ����ģ�͵Ĵ�С�ͼ�����
model = model.module.to(device)
model_size = get_model_size(model)
tflops = get_tflops(model_size, batch_size, 1024, 1.0)  # ����ÿ����ʱ1��

# ��rankΪ0�Ľ����ϴ�ӡģ�ʹ�С�ͼ�����
if rank == 0:
    print(f"Estimated Model Size: {model_size} parameters")
    print(f"Estimated TFLOPs: {tflops:.2f}")
