#encoding: utf-8
# �����࿨DPѵ��
#torchrun --nnodes 1 --nproc_per_node=4 train_dp.py 
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import GPTLMModel, GPTLMLoss, get_dataloader, get_model_size, get_tflops
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# ����ѵ������
batch_size = 2
learning_rate = 5e-4
num_epochs = 10
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ����ģ�ͺ���ʧ����
model = GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, max_seq_len=1024)
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.to(device)
model = DDP(model, device_ids=None)
loss_fn = GPTLMLoss().to(device)

# �������ݼ�����
dataloader = get_dataloader(vocab_size=50257, seq_length=1024, batch_size=batch_size, data_size=256, num_workers=4)

# ѡ���豸
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dist.init_process_group(backend='nccl', init_method='env://', world_size=2, rank=0)
# ѡ���Ż���
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = nn.SyncBatchNorm.convert_sync_batchnorm(optimizer)
optimizer = DDP(optimizer, device_ids=None)
# ѵ��ѭ��
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        input_ids = batch[0]
        attention_mask = torch.ones_like(input_ids)
        logits = model(input_ids, attention_mask)
        
        optimizer.zero_grad()
        loss = loss_fn(logits, input_ids)  # ������ʧ
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")

# ����ģ�͵Ĵ�С�ͼ�����
model_size = get_model_size(model.module)
tflops = get_tflops(model_size, batch_size, 1024, 1.0)  # ����ÿ����ʱ1��
print(f"Estimated Model Size: {model_size} parameters")
print(f"Estimated TFLOPs: {tflops:.2f}")

dist.destroy_process_group()
