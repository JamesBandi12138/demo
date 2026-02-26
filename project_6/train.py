import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

base_path = r'e:\AIAgent\GIT\demo\project_6'
train_label_file = os.path.join(base_path, '初赛训练数据 2019-08-01', 'TRAIN', 'Train_labels.csv')
train_review_file = os.path.join(base_path, '初赛训练数据 2019-08-01', 'TRAIN', 'Train_reviews.csv')
test_review_file = os.path.join(base_path, '初赛测试数据 2019-08-15', 'TEST', 'Test_reviews.csv')
output_file = os.path.join(base_path, 'Result.csv')

CATEGORIES = ['整体', '使用体验', '功效', '价格', '物流', '气味', '包装', '真伪', '服务', '其他', '成分', '尺寸', '新鲜度']
POLARITIES = ['正面', '负面', '中性']
cat2idx = {c: i for i, c in enumerate(CATEGORIES)}
idx2cat = {i: c for i, c in enumerate(CATEGORIES)}
pol2idx = {p: i for i, p in enumerate(POLARITIES)}
idx2pol = {i: p for i, p in enumerate(POLARITIES)}

print('读数据...')
labels = pd.read_csv(train_label_file)
reviews = pd.read_csv(train_review_file)
review_dict = dict(zip(reviews['id'], reviews['Reviews']))

# 从训练数据里把观点词和属性词都扒下来
opinion_words = []
aspect_words = []
opinion_info = {}
aspect_info = {}

for _, row in labels.iterrows():
    cat = row['Categories']
    pol = row['Polarities']
    asp = str(row['AspectTerms']).strip()
    opi = str(row['OpinionTerms']).strip()
    
    if asp and asp != '_':
        if asp not in aspect_info:
            aspect_info[asp] = {'cat': Counter(), 'pol': Counter()}
            aspect_words.append(asp)
        aspect_info[asp]['cat'][cat] += 1
        aspect_info[asp]['pol'][pol] += 1
    
    if opi and opi != '_':
        if opi not in opinion_info:
            opinion_info[opi] = {'cat': Counter(), 'pol': Counter()}
            opinion_words.append(opi)
        opinion_info[opi]['cat'][cat] += 1
        opinion_info[opi]['pol'][pol] += 1

# 按频率排个序，常用的放前面
opinion_words.sort(key=lambda x: -sum(opinion_info[x]['cat'].values()))
aspect_words.sort(key=lambda x: -sum(aspect_info[x]['cat'].values()))
print(f'属性词{len(aspect_words)}个, 观点词{len(opinion_words)}个')

# 字符表
char2idx = {'<PAD>': 0, '<UNK>': 1}
for _, row in labels.iterrows():
    txt = str(review_dict.get(row['id'], ''))
    for ch in txt:
        if ch not in char2idx:
            char2idx[ch] = len(char2idx)
print(f'字符表{len(char2idx)}个')

def text2ids(text, max_len=128):
    ids = [char2idx.get(c, 1) for c in text]
    real_len = min(len(ids), max_len)
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids, real_len

# 准备训练数据
print('准备数据...')
X = []
y_cat = []
y_pol = []

for _, row in labels.iterrows():
    txt = str(review_dict.get(row['id'], ''))
    txt = re.sub(r'<[^>]+>', '', txt).strip()
    ids, _ = text2ids(txt)
    X.append(ids)
    y_cat.append(cat2idx.get(row['Categories'], 0))
    y_pol.append(pol2idx.get(row['Polarities'], 0))

X = torch.tensor(X, dtype=torch.long)
y_cat = torch.tensor(y_cat, dtype=torch.long)
y_pol = torch.tensor(y_pol, dtype=torch.long)

# 划分
n = len(X)
idx = np.random.permutation(n)
split = int(n * 0.9)
train_idx, val_idx = idx[:split], idx[split:]

# 模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(char2idx), 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 256, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc_cat = nn.Linear(512, len(CATEGORIES))
        self.fc_pol = nn.Linear(512, len(POLARITIES))
        self.drop = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.drop(self.emb(x))
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h = self.drop(h)
        return self.fc_cat(h), self.fc_pol(h)

# 训练
print('训练...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 64
epochs = 15
best_acc = 0

train_X = X[train_idx]
train_y1 = y_cat[train_idx]
train_y2 = y_pol[train_idx]
val_X = X[val_idx]
val_y1 = y_cat[val_idx]
val_y2 = y_pol[val_idx]

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for i in range(0, len(train_X), batch_size):
        bx = train_X[i:i+batch_size].to(device)
        by1 = train_y1[i:i+batch_size].to(device)
        by2 = train_y2[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        o1, o2 = model(bx)
        loss = criterion(o1, by1) + criterion(o2, by2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        vo1, vo2 = model(val_X.to(device))
        pred_cat = vo1.argmax(dim=1).cpu()
        pred_pol = vo2.argmax(dim=1).cpu()
        acc_cat = (pred_cat == val_y1).float().mean().item()
        acc_pol = (pred_pol == val_y2).float().mean().item()
    
    print(f'epoch {epoch+1}, loss:{total_loss:.2f}, cat_acc:{acc_cat:.4f}, pol_acc:{acc_pol:.4f}')
    
    if acc_cat + acc_pol > best_acc:
        best_acc = acc_cat + acc_pol
        torch.save({'state_dict': model.state_dict()}, os.path.join(base_path, 'model.pt'))

# 预测
print('预测...')
test_df = pd.read_csv(test_review_file)
results = []

ckpt = torch.load(os.path.join(base_path, 'model.pt'), weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()

def find_terms(text):
    found_asp = []
    found_opi = []
    for a in aspect_words:
        if a in text:
            found_asp.append(a)
    for o in opinion_words:
        if o in text:
            found_opi.append(o)
    return found_asp[:2], found_opi[:2]

with torch.no_grad():
    for _, row in test_df.iterrows():
        txt = str(row['Reviews'])
        txt = re.sub(r'<[^>]+>', '', txt).strip()
        rid = row['id']
        
        ids, _ = text2ids(txt)
        inp = torch.tensor([ids], dtype=torch.long).to(device)
        
        o1, o2 = model(inp)
        pred_c = o1.argmax(dim=1).item()
        pred_p = o2.argmax(dim=1).item()
        
        cat = idx2cat[pred_c]
        pol = idx2pol[pred_p]
        
        asps, opis = find_terms(txt)
        
        if not asps and not opis:
            results.append([rid, '_', '_', '_', '_'])
        else:
            n = max(len(asps), len(opis))
            for i in range(n):
                a = asps[i] if i < len(asps) else '_'
                o = opis[i] if i < len(opis) else '_'
                
                if a == '_' and o == '_':
                    continue
                
                if o != '_':
                    word_cat = opinion_info[o]['cat'].most_common(1)[0][0]
                    word_pol = opinion_info[o]['pol'].most_common(1)[0][0]
                else:
                    word_cat = aspect_info[a]['cat'].most_common(1)[0][0]
                    word_pol = aspect_info[a]['pol'].most_common(1)[0][0]
                
                results.append([rid, a, o, word_cat, word_pol])

# 保存
df = pd.DataFrame(results)
df = df.sort_values(0)
df.to_csv(output_file, index=False, header=False, encoding='utf-8')
print(f'完成, 共{len(df)}条')
