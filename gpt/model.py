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
        # 只返回模型的预测值，即logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]

class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        #创建了一个交叉熵损失函数对象，用于计算模型的损失

    def forward(self, logits, labels):
        # logits: [batch_size, seq_len, vocab_size] logits是模型的预测值
        # labels: [batch_size, seq_len] labels是真实值
        # Shift so that tokens < n predict n
        # 将logits和labels的最后一个维度切片，即序列长度维度，
        # 从而将序列中的每个token的预测值和标签值对齐
        # 例如，logits的形状为[8, 1024, 50257]，labels的形状为[8, 1024]，
        # 则切片后的logits的形状为[8, 1023, 50257]，labels的形状为[8, 1023]，
        # 即logits的第一个token的预测值和labels的第一个token的标签值对齐，
        # logits的第二个token的预测值和labels的第二个token的标签值对齐，
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # 将切片后的logits和labels的最后一个维度展平，即序列长度维度
        # 例如，logits的形状为[8, 1023, 50257]，labels的形状为[8, 1023]，
        # 则展平后的logits的形状为[8192, 50257]，labels的形状为[8192]，

        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#使用DataLoader将数据集、批大小、工作线程数、
# 是否固定内存等参数传递给数据加载器，返回数据加载器对象
def get_dataloader(
    vocab_size: int = 50257, #词汇表大小
    seq_length: int = 1024, #序列长度
    batch_size: int = 8, #批大小
    data_size: int = 256, #数据集大小
    num_workers: int = 4, #工作线程数
    pin_memory: bool = True, #是否将数据加载到固定内存中
    use_distributed_sampler: bool = False #是否使用分布式采样器
):
    ids = torch.randint(vocab_size, (data_size, seq_length))
    #使用torch.randint创建一个随机的数据集ids，
    #其值在词汇表大小范围内，并且具有指定的data_size和seq_length
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
    #计算模型的浮点运算次数（TFLOPs），接受模型的参数数量、批大小、序列长度以及步骤时间作为输入
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)

def get_model_size(model: nn.Module):
    #计算模型的参数数量，接受模型对象作为输入
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel  #所有模块的参数数量总和作为结果返回
