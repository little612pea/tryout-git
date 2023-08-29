#encoding:utf-8
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler

class GPTLMModel(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 max_seq_len=1024,
                 vocab_size=50257,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.config = GPT2Config(n_embd=hidden_size,
                                 n_layer=num_layers,
                                 n_head=num_attention_heads,
                                 n_positions=max_seq_len,
                                 n_ctx=max_seq_len,
                                 vocab_size=vocab_size)
        self.model = GPT2LMHeadModel(self.config)
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # ֻ����ģ�͵�Ԥ��ֵ����logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]

class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        #������һ����������ʧ�����������ڼ���ģ�͵���ʧ

    def forward(self, logits, labels):
        # logits: [batch_size, seq_len, vocab_size] logits��ģ�͵�Ԥ��ֵ
        # labels: [batch_size, seq_len] labels����ʵֵ
        # Shift so that tokens < n predict n
        # ��logits��labels�����һ��ά����Ƭ�������г���ά�ȣ�
        # �Ӷ��������е�ÿ��token��Ԥ��ֵ�ͱ�ǩֵ����
        # ���磬logits����״Ϊ[8, 1024, 50257]��labels����״Ϊ[8, 1024]��
        # ����Ƭ���logits����״Ϊ[8, 1023, 50257]��labels����״Ϊ[8, 1023]��
        # ��logits�ĵ�һ��token��Ԥ��ֵ��labels�ĵ�һ��token�ı�ǩֵ���룬
        # logits�ĵڶ���token��Ԥ��ֵ��labels�ĵڶ���token�ı�ǩֵ���룬
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # ����Ƭ���logits��labels�����һ��ά��չƽ�������г���ά��
        # ���磬logits����״Ϊ[8, 1023, 50257]��labels����״Ϊ[8, 1023]��
        # ��չƽ���logits����״Ϊ[8192, 50257]��labels����״Ϊ[8192]��

        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#ʹ��DataLoader�����ݼ�������С�������߳�����
# �Ƿ�̶��ڴ�Ȳ������ݸ����ݼ��������������ݼ���������
def get_dataloader(
    vocab_size: int = 50257, #�ʻ���С
    seq_length: int = 1024, #���г���
    batch_size: int = 8, #����С
    data_size: int = 256, #���ݼ���С
    num_workers: int = 4, #�����߳���
    pin_memory: bool = True, #�Ƿ����ݼ��ص��̶��ڴ���
    use_distributed_sampler: bool = False #�Ƿ�ʹ�÷ֲ�ʽ������
):
    ids = torch.randint(vocab_size, (data_size, seq_length))
    #ʹ��torch.randint����һ����������ݼ�ids��
    #��ֵ�ڴʻ���С��Χ�ڣ����Ҿ���ָ����data_size��seq_length
    dataset = TensorDataset(ids)
    if use_distributed_sampler:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler
    )



def get_tflops(model_numel, batch_size, seq_len, step_time):
    #����ģ�͵ĸ������������TFLOPs��������ģ�͵Ĳ�������������С�����г����Լ�����ʱ����Ϊ����
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)

def get_model_size(model: nn.Module):
    #����ģ�͵Ĳ�������������ģ�Ͷ�����Ϊ����
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel  #����ģ��Ĳ��������ܺ���Ϊ�������
