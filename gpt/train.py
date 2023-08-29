#encoding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import GPTLMModel, GPTLMLoss, get_dataloader, get_model_size, get_tflops

# ����ѵ������
batch_size = 2
learning_rate = 5e-4
num_epochs = 10

# ����ģ�ͺ���ʧ����
model = GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, max_seq_len=1024)
loss_fn = GPTLMLoss()

# �������ݼ�����
dataloader = get_dataloader(vocab_size=50257, seq_length=1024, batch_size=batch_size, data_size=256, num_workers=4)

# ѡ���豸
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ѡ���Ż���
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ѵ��ѭ��
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        input_ids = batch[0].to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        logits = model(input_ids, attention_mask)
        
        optimizer.zero_grad()
        loss = loss_fn(logits, input_ids)  # ������ʧ
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")

# ����ģ�͵Ĵ�С�ͼ�����
model_size = get_model_size(model)
tflops = get_tflops(model_size, batch_size, 1024, 1.0)  # ����ÿ����ʱ1��
print(f"Estimated Model Size: {model_size} parameters")
print(f"Estimated TFLOPs: {tflops:.2f}")
