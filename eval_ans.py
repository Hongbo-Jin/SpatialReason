import json

choices=['(A)','(B)','(C)','(D)','(E)']
choices2=['A','B','C','D','E']

preds = [json.loads(q) for q in open('/mnt/cloud_disk/jhb/binjiang/SpatialReason/pred/qwen2.5vl_7B_scene0_ans.json', "r")]

acc=0
total=0
err=0

for pred in preds:
    pred_answer=pred['pred_answer'][0][0:3]
    gt_answer=pred['answer']
    
    if pred_answer in choices:
        idx=choices.index(pred_answer)
        pred_answer=choices2[idx]
    else:
        err+=1
    
    if pred_answer==gt_answer:
        acc+=1
    total+=1

print(f'total accuracy: {acc/total}')
print(f'err :{err}')
    
