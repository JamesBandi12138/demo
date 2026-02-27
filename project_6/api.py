import re
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import os

app = FastAPI(title='电商评论观点挖掘API')
base_path = os.path.dirname(os.path.abspath(__file__))

CATEGORY_LIST = ['整体', '使用体验', '功效', '价格', '物流', '气味', '包装', '真伪', '服务', '其他', '成分', '尺寸', '新鲜度']
POLARITY_LIST = ['正面', '负面', '中性']

category_to_id = {c: i for i, c in enumerate(CATEGORY_LIST)}
id_to_category = {i: c for i, c in enumerate(CATEGORY_LIST)}
polarity_to_id = {p: i for i, p in enumerate(POLARITY_LIST)}
id_to_polarity = {i: p for i, p in enumerate(POLARITY_LIST)}

opinion_word_list = []
aspect_word_list = []
opinion_word_info = {}
aspect_word_info = {}
char_to_id = {}
model = None
device = None

class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.嵌入层 = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 256, batch_first=True, bidirectional=True, dropout=0.3)
        self.类别输出 = nn.Linear(512, len(CATEGORY_LIST))
        self.极性输出 = nn.Linear(512, len(POLARITY_LIST))
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.dropout(self.嵌入层(x))
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        return self.类别输出(hidden), self.极性输出(hidden)

class PredictRequest(BaseModel):
    text: str

class PredictResult(BaseModel):
    aspect: str
    opinion: str
    category: str
    polarity: str

class PredictResponse(BaseModel):
    results: List[PredictResult]

def load_model():
    global opinion_word_list, aspect_word_list, opinion_word_info, aspect_word_info, char_to_id, model, device
    
    train_label_file = os.path.join(base_path, '初赛训练数据 2019-08-01', 'TRAIN', 'Train_labels.csv')
    train_review_file = os.path.join(base_path, '初赛训练数据 2019-08-01', 'TRAIN', 'Train_reviews.csv')
    
    labels = pd.read_csv(train_label_file)
    reviews = pd.read_csv(train_review_file)
    review_dict = dict(zip(reviews['id'], reviews['Reviews']))
    
    opinion_word_list = []
    aspect_word_list = []
    opinion_word_info = {}
    aspect_word_info = {}
    
    for _, row in labels.iterrows():
        category = row['Categories']
        polarity = row['Polarities']
        aspect = str(row['AspectTerms']).strip()
        opinion = str(row['OpinionTerms']).strip()
        
        if aspect and aspect != '_':
            if aspect not in aspect_word_info:
                aspect_word_info[aspect] = {'category': Counter(), 'polarity': Counter()}
                aspect_word_list.append(aspect)
            aspect_word_info[aspect]['category'][category] += 1
            aspect_word_info[aspect]['polarity'][polarity] += 1
        
        if opinion and opinion != '_':
            if opinion not in opinion_word_info:
                opinion_word_info[opinion] = {'category': Counter(), 'polarity': Counter()}
                opinion_word_list.append(opinion)
            opinion_word_info[opinion]['category'][category] += 1
            opinion_word_info[opinion]['polarity'][polarity] += 1
    
    opinion_word_list.sort(key=lambda x: -sum(opinion_word_info[x]['category'].values()))
    aspect_word_list.sort(key=lambda x: -sum(aspect_word_info[x]['category'].values()))
    
    char_to_id = {'<PAD>': 0, '<UNK>': 1}
    for _, row in labels.iterrows():
        text = str(review_dict.get(row['id'], ''))
        for char in text:
            if char not in char_to_id:
                char_to_id[char] = len(char_to_id)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(len(char_to_id)).to(device)
    checkpoint = torch.load(os.path.join(base_path, 'model.pt'), weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['模型参数'])
    model.eval()
    print('模型加载完成')

def text_to_ids(text, max_len=128):
    ids = [char_to_id.get(c, 1) for c in text]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

def find_aspect_and_opinion(text):
    found_aspect = []
    found_opinion = []
    for word in aspect_word_list:
        if word in text:
            found_aspect.append(word)
    for word in opinion_word_list:
        if word in text:
            found_opinion.append(word)
    return found_aspect[:2], found_opinion[:2]

def predict(text):
    text = re.sub(r'<[^>]+>', '', text).strip()
    
    ids = text_to_ids(text)
    input_tensor = torch.tensor([ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        category_output, polarity_output = model(input_tensor)
        pred_category_id = category_output.argmax(dim=1).item()
        pred_polarity_id = polarity_output.argmax(dim=1).item()
    
    pred_category = id_to_category[pred_category_id]
    pred_polarity = id_to_polarity[pred_polarity_id]
    
    aspects, opinions = find_aspect_and_opinion(text)
    
    results = []
    if not aspects and not opinions:
        results.append({
            'aspect': '_', 
            'opinion': '_', 
            'category': '_', 
            'polarity': '_'
        })
    else:
        count = max(len(aspects), len(opinions))
        for i in range(count):
            aspect = aspects[i] if i < len(aspects) else '_'
            opinion = opinions[i] if i < len(opinions) else '_'
            
            if aspect == '_' and opinion == '_':
                continue
            
            if opinion != '_':
                word_category = opinion_word_info[opinion]['category'].most_common(1)[0][0]
                word_polarity = opinion_word_info[opinion]['polarity'].most_common(1)[0][0]
            else:
                word_category = aspect_word_info[aspect]['category'].most_common(1)[0][0]
                word_polarity = aspect_word_info[aspect]['polarity'].most_common(1)[0][0]
            
            results.append({
                'aspect': aspect,
                'opinion': opinion,
                'category': word_category,
                'polarity': word_polarity
            })
    
    return results

@app.on_event("startup")
async def startup():
    load_model()

@app.post('/predict', response_model=PredictResponse)
async def api_predict(request: PredictRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail='请提供text参数')
    
    results = predict(request.text)
    return {'results': results}

@app.get('/health')
async def health():
    return {'status': 'ok'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
