import torch

files = [
    r'e:\AIAgent\GIT\demo\project_6\model.pt',
    r'e:\AIAgent\GIT\demo\project_6\cls_model.pt',
    r'e:\AIAgent\GIT\demo\project_6\ner_model.pt'
]

for f in files:
    print(f'=== {f.split(chr(92))[-1]} ===')
    ckpt = torch.load(f, map_location='cpu', weights_only=False)
    print(f'Keys: {ckpt.keys()}')
    if 'vocab' in ckpt:
        print(f'Vocab size: {len(ckpt["vocab"])}')
    print()
