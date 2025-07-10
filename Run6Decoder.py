"""
    使用TransformerDecoder生成IPv6地址
 
    版本说明：
        v5.5：在v.4.1上进行调整
        v5.6：在v5.5的基础上增加记录中间结果
        v5.7：增加模型参数的带入
        v5.8：增加显示模型的参数量
        v5.9：生成地址时，使用ordered_set库，即保持生成地址的顺序，又可去重
              并将生成的地址按先后每1M保存一个文件
        v5.10，在生成地址的过程中保存到文件，而不是全部生成完成后再分割保存。
"""

__version = 'v5.10'

import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm
import random
import sys
import math
import numpy as np
from datetime import datetime
from Transformer import TransformerDecoder
from ordered_set import OrderedSet


# 训练相关参数
DATA_FILE = '../data/paper_data/data_sets/C3_Down_1M_32hex.txt'
MODEL_FILE  = 'data/model6decoder.pth'
GEN_ADDR_FILE = '/data/targets.txt'
MAX_LEN = 34    # <bos> + 32个字符 + <eos>
BATCH_SIZE = 64
DATA_SHUFFLE = True
EPOCH_NUM = 5
LEARNING_RATE=5e-5
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


# 模型相关参数
N_LAYER = 6
N_HEAD = 8
FEED_FORWARD_DIM = 2048
EMBEDDING_DIM = 512    # d_model
DROPOUT = 0.1
TOPK = 5     # 生成时取概率最高的k个字



# tonken相关变量
BOS, EOS = '<bos>', '<eos>'
tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', BOS, EOS]
token_to_id = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
               'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15, BOS:16, EOS:17}
id_to_token = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
               10:'a', 11:'b', 12:'c', 13:'d', 14:'e', 15:'f', 16:BOS, 17:EOS}
BOS_ID = token_to_id[BOS]
EOS_ID = token_to_id[EOS]
VOCAB_SIZE = len(tokens)



def token_encode(tokens):
    """ 词列表 -> <bos>编号 + 编号列表 + <eos>编号 """
    token_ids = [BOS_ID] # 起始标记
    # 遍历，词转编号
    for token in tokens:
        token_ids.append(token_to_id[token])
    token_ids.append(EOS_ID) # 结束标记
    return token_ids



def token_decode(token_ids):
    """ 编号列表 -> 词列表(去掉起始、结束标记) """
    tokens = []
    for idx in token_ids:
        # 跳过起始、结束标记
        if idx != BOS_ID and idx != EOS_ID:
            tokens.append(id_to_token[idx])
    return tokens    


class IPv6AddrSet(Dataset):
    """ 自定义IPv6地址数据集类 """
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    

def load_data(seed_file=DATA_FILE, batch_size=BATCH_SIZE):
    """ 读取IPv6地址数据文件，并返回dataloader """
    with open(seed_file, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    # 将一IPv6地址转换为长度为32的整数(0-15)列表，并加上<bos>, <eos>
    address = []
    for line in raw_data:
        address.append(token_encode(line.strip()))
    dataset = IPv6AddrSet(np.array(address))
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=DATA_SHUFFLE)
    return dataloader




class Model6Decoder(nn.Module):
    """
    定义IPv6 TransformerDecoder 模型
    """
    def __init__(self, dict_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, nhead=N_HEAD,
                 dim_feedforward=FEED_FORWARD_DIM, num_layers=N_LAYER, dropout=DROPOUT, 
                 activation=F.gelu):
        super(Model6Decoder, self).__init__()
        # Embedding层
        self.embedding = nn.Embedding(num_embeddings=dict_size, embedding_dim=embedding_dim)
        norm = nn.LayerNorm(embedding_dim)
        self.decoder = TransformerDecoder(d_model=embedding_dim, nhead=nhead, dropout=dropout,
                                          dim_feedforward=dim_feedforward, num_layers=num_layers,
                                          norm=norm, activation=activation)
        # TransformerDecoder最后一个参数norm非常重要，它是在所有DecoderLayer处理完成后，再做一次AddNorm
        # 实验发现，如果少最后一次AddNor，命中率会从35%降至25%
        
        # 线性层输出需要和原始词典的字符编号范围对应
        self.predictor = nn.Linear(embedding_dim, dict_size)

    def forward(self, tgt, device=DEVICE):
        # tgt是经过“词 -> 编号”的转换，但未经嵌入和位置编码PE

        # 生成 Mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(device)

        # 词嵌入
        tgt = self.embedding(tgt)

        # permute(1, 0, 2) 将src切换成“批次”在中间维度的形式，因为没有设置batch_first
        out = self.decoder(tgt.permute(1, 0, 2), tgt_mask=tgt_mask)
        out = self.predictor(out)

        return out



def linear_map(values, in_min, in_max, out_min, out_max):
    """
    使用numpy进行批量线性映射   
    参数:
        values: numpy数组或可迭代的数值
        其他参数同linear_map        
    返回:
        映射后的numpy数组
    """
    values = np.asarray(values)
    in_span = in_max - in_min
    out_span = out_max - out_min
    scaled = (values - in_min) / in_span
    return out_min + (scaled * out_span)




def train_model(model, seed_file=DATA_FILE, model_file=None,
                batch_size=BATCH_SIZE, lr=LEARNING_RATE, epochs=EPOCH_NUM, 
                device=DEVICE, budget=100000, res_num=1):
    """ 模型训练 """

    # 计算保存中间结果的epoch
    if res_num > 1:
        outs = np.array(range(1, res_num+1))
        mapped_values = linear_map(outs, 1, res_num, 1, epochs)
        # epoch_save是将要保存中间结果的epoch列表
        epoch_save = np.round(mapped_values).astype(int)
     
    dataloader = load_data(seed_file, batch_size) # 加载数据 

    # 模型损失函数和优化器 
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        data_progress = tqdm(dataloader, desc="Train...")
        for step, data in enumerate(data_progress, start=1):
            data = data.to(device)

            # 构造输入与标签
            tgt = data[:, :-1]
            tgt_y = data[:, 1:]
            
            # 进行Transformer的计算，再将结果送给最后的线性层进行预测
            # out与tgt的token数相等，最好状态是tgt向后错位一个token
            out = model(tgt, device)
            loss = criteria(out.permute(1,2,0).contiguous(), tgt_y.to(dtype=torch.long))           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            total_loss += loss.item()

            # 更新训练进度
            data_progress.set_description(f"Train... [epoch {epoch}/{epochs}, loss {(total_loss / step):.5f}]")

        # 保存模型训练过程参数
        if res_num > 1:
            if epoch == epoch_save[0]:
                # 此将处epoch_save用作队列，保存一次中间结果后，将队列头删除
                epoch_save = np.delete(epoch_save, 0)
                if model_file is not None:
                    model_file_epoch = f'{model_file[:-4]}_epoch_{epoch}_loss_{total_loss/step}.path'
                    torch.save(model.state_dict(), model_file_epoch)
                    # 生成中间结果
                    target_file_epoch = f'{model_file[:-4]}_epoch_{epoch}_loss_{total_loss/step}_target.txt'
                    generate_target(model=model, top_k=TOPK, budget=budget, 
                            target_file=target_file_epoch, batch_size=2048, device=device)

    # 保存最后的训练结果
    if res_num==1 and model_file is not None:
        torch.save(model.state_dict(), model_file)
        
    # 返回最终的训练平均误差
    return total_loss / step




def ids_to_ipv6(addr):
    ''' 将32位token列表转换为ipv6地址 '''
    ipv6 = ''
    for i in range(len(addr)):
        ipv6 += id_to_token[addr[i]]
        if i%4 == 3 and i < 31:
            ipv6 += ':'
    return ipv6



def gen_addr_batch(model, top_k, head_num, head_batch, device=DEVICE):
    """ 生成一个批次IPv6地址 """
    with torch.no_grad():
        #word_ids = token_encode(text)
        # 将IPv6地址头转换为tensor
        head_batch = torch.tensor(head_batch, dtype=torch.long, device=device)

        # 去掉最后的,<eos>标志
        tgt = head_batch[:,:-1]

        i = 0
        while i < 32 - head_num:
            # 前向传播，out.shape=(sequence_len, batch_size, embed_dim)
            out = model(tgt, device)

             # 预测结果分类（0-f,bos,eos）不要最后的EOS、BOS
            _probas = out[-1, :, :-2]

            # 将小于topk中最小值的概率全部置为-∞
            indices_to_remove = _probas < torch.topk(_probas, top_k)[0][..., -1, None]
            _probas[indices_to_remove] = -float('Inf')

            # softmax操作，以使概率高的更容易被选择
            _probas = F.softmax(_probas, dim=-1)

            # 在topk中根据概率高低选择一个
            y = torch.multinomial(_probas, num_samples=1)
            
            # 和之前的预测结果拼接到一起
            tgt = torch.cat((tgt, y), dim=-1)
            i += 1

        #result = torch.cat((src, tgt), dim=-1)
        ipv6list = list(map(ids_to_ipv6, tgt[:, 1:].tolist()))
        return ipv6list



def generate_target(model, top_k, budget, target_file, batch_size=BATCH_SIZE, device=DEVICE, head='2'):
    ''' 生成一定量的IPv6地址，并写入文件'''
    
    head_num = len(head)   # 地址头长度
    
    # 对IPv6地址头进行编码
    head_tokens_ids = token_encode(head)

    # 将单个地址头复制为批处理模式，并转换为tensor
    head_batch = [head_tokens_ids for i in range(batch_size)]
    
    model.eval() # 模型切换到推理模式

    # 按批次生成IPv6地址
    curr_seg = 1    # 当前将分割文件序号
    curr_file = target_file[:-4]+f'_{curr_seg:02d}M.txt'  # 将分割文件名
    addrs = OrderedSet()    # 暂存生成地址的有序集合变量
    progress_bar = tqdm(total=budget, desc="Generating...") # 显示进度条
    while len(addrs) < budget:
        gen_addr = gen_addr_batch(model, top_k, head_num, head_batch, device=device)
        # 将每个IPv6加个换行符
        addrn = list(map(lambda s: s + "\n", gen_addr))  # 地址后加换行符
        addrs.update(addrn)
        progress_bar.n = len(addrs)  # 设置进度条
        progress_bar.refresh()    # 手动刷新进度条显示
        
        # 若需要文件分割，则在生成地址过程中将地址保存到文件
        if budget > 1000000:
            # 检查是否达到1M大小
            if len(addrs) >= curr_seg*1000000:
                with open(curr_file, 'w') as f:
                    f.writelines(addrs[(curr_seg-1)*1000000:curr_seg*1000000])
                curr_seg += 1
                curr_file = target_file[:-4]+f'_{curr_seg:02d}M.txt'
    # 所有地址全部成生
    if budget > 1000000:
        # 保存不够1M的余下地址
        if budget > (curr_seg-1)*1000000: 
            with open(curr_file, 'w') as f:
                f.writelines(addrs[(curr_seg-1)*1000000:budget])
    else:
        # 对于无需分割的，保存为一个文件
        with open(target_file, 'w') as f:
            f.writelines(addrs[:budget])
     




if __name__ == '__main__':
    '''
    程序运行示例：
    python Run6Decoder.py --seed_file=../data/paper_data/data_sets/C2_Down_100K_32hex.txt \
                          --model_file=data/model_C2_Down_100K_bs16_ep90_lr5e5.pth \
                          --target_file=data/Gen_C2_Down_100K_bs16_ep90_lr5e5.txt \
                          --batch_size=16 \
                          --epochs=90 \
                          --learning_rate=5e-5 \
                          --top_k=5 \
                          --device=cuda:1 \
                          --budget=100000 \
                          --res_num=1
    '''

    # 处理程序带入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_train', action='store_true', default=False, help='无需重新训练标志')
    parser.add_argument('--seed_file', default=DATA_FILE, type=str, required=False, help='待训练的IPv6地址集文件名')
    parser.add_argument('--model_file', default=MODEL_FILE, type=str, required=False, help='待训练的IPv6地址集文件名')
    parser.add_argument('--target_file', default=GEN_ADDR_FILE, type=str, required=False, help='生成候选IPv6地址文件名')
    parser.add_argument('--n_layer', default=N_LAYER, type=int, required=False, help='模型层数')
    parser.add_argument('--n_head', default=N_HEAD, type=int, required=False, help='抽头数量')
    parser.add_argument('--ff_dim', default=FEED_FORWARD_DIM, type=int, required=False, help='前馈网络维度')
    parser.add_argument('--embed_dim', default=EMBEDDING_DIM, type=int, required=False, help='数据嵌入维度')
    parser.add_argument('--dropout', default=DROPOUT, type=float, required=False, help='dropout层丢率率')
    parser.add_argument('--top_k', default=TOPK, type=int, required=False, help='生成地址时在概率最高n个中选取')
    parser.add_argument('--epochs', default=EPOCH_NUM, type=int, required=False, help='训练循环次数')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, required=False, help='模型处理的批次大小')
    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float, required=False, help='模型训练时的学习率') 
    parser.add_argument('--budget', default=1000000, type=int, required=False, help='单次生成地址数量')   
    parser.add_argument('--device', default=DEVICE, type=str, required=False, help='训练和推理设备')
    parser.add_argument('--res_num', default=1, type=int, required=False, help='训练时保存结果数量，若大于1，则保存中间结果')
    args = parser.parse_args()

    # 实例化模型
    model = Model6Decoder(embedding_dim=args.embed_dim, nhead=args.n_head, dim_feedforward=args.ff_dim, num_layers=args.n_layer, dropout=args.dropout).to(args.device)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # 模型训练
    if args.no_train:
        model.load_state_dict(torch.load(args.model_file, map_location=args.device, weights_only=True))  # 读入模型参数
    else:
        train_model(model=model, seed_file=args.seed_file, model_file=args.model_file,
                batch_size=args.batch_size, lr=args.learning_rate, epochs=args.epochs, 
                device=args.device, budget=args.budget, res_num=args.res_num)          

    # 如果不重新训练或者重新训练不保存中间结果，则生成IPv6地址
    if args.no_train or args.res_num == 1:
        generate_target(model=model, top_k=args.top_k, budget=args.budget, 
                        target_file=args.target_file, batch_size=2048, device=args.device)

