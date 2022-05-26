# packages
import os
import numpy as np 
import json
import argparse

def get_config():
    """ Set default and command line arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True, type=str, )
    parser.add_argument("--epoch", required=True, type=int)
    args = parser.parse_args()
    return args 


def obtain_qid2sqids(questions,sub_questions):
    qid2sqids = {}
    for qid in questions:
        qid2sqids[qid] = []
    for s_qid in sub_questions:
        qid, q_count = s_qid.split('_')
        qid2sqids[qid].append(s_qid)
    return qid2sqids

def load_pred_files(exp_name,epoch,eval_split,res_root):
    
    epoch = '%04d' % epoch if epoch != 'best' else epoch
    exp_name_epoch = exp_name + '_' + epoch
    res_dir = os.path.join(res_root,exp_name,epoch)
    
    pred_path = os.path.join(res_dir,'pred_' + exp_name_epoch + '_' + eval_split + '_balanced.json')
    if os.path.exists(pred_path):
        preds_raw = json.load(open(pred_path,'r'))
        preds = {x['questionId']:x for x in preds_raw}
        print(len(preds),pred_path+' is loaded')
    else:
        print(pred_path+' does not exist')
        preds = None
        
    preds_sub_path = os.path.join(res_dir,'pred_' + exp_name_epoch + '_' + eval_split + '_sub_balanced.json')
    if os.path.exists(preds_sub_path):
        preds_sub_raw = json.load(open(preds_sub_path,'r'))
        preds_sub = {x['questionId']:x for x in preds_sub_raw}
        print(len(preds_sub),preds_sub_path+' is loaded')
    else:
        print(preds_sub_path+' does not exist')
        preds_sub = None
    
    res = {}
    res['split'] = eval_split
    res['preds'], res['preds_sub'] = preds, preds_sub
    return res 

def obtain_consistency_from_res(res,questions,qid2sqids,split='val'):
    
    assert res['split'] == split
    preds = res['preds']
    preds_sub = res['preds_sub']
    questions_ori = questions[split]
    questions_sub = questions[split+'_sub']
    
    count = 0
    correct = 0
    counts_sub = np.zeros([3]) 
    corrects_sub = np.zeros([3]) 

    for qid in questions_ori:
        count += 1
        q = questions_ori[qid]
        sq_num = len(qid2sqids[qid])
        if q['answer'] == preds[qid]['prediction']:
            correct += 1
            flag = 0 

            for sqid in qid2sqids[qid]:
                sq = questions_sub[sqid]
                if sq['answer'] != preds_sub[sqid]['prediction']:
                    flag = 1
            
            for i in range(3):
                if i < sq_num:
                    counts_sub[i] += 1
                    if flag == 0:
                        corrects_sub[i] += 1

    consistency = corrects_sub/counts_sub
    print('rc_k',consistency)


def main():
    
    args = get_config()

    question_root = 'exp/dataset/questions'
    results_root = 'exp/results'

    questions = {}
    val_questions = json.load(open(os.path.join(question_root,'val_balanced_questions.json')))
    questions['val'] = val_questions

    val_sub_questions = json.load(open(os.path.join(question_root,'val_sub_balanced_questions.json')))
    questions['val_sub'] = val_sub_questions

    qid2sqids_val = obtain_qid2sqids(val_questions,val_sub_questions)

    res = load_pred_files(args.exp_name,args.epoch,'val',results_root) 
    obtain_consistency_from_res(res,questions,qid2sqids_val)


if __name__ == "__main__":
    main()