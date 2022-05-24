import os
import json
import torch
import sys

sys.path.append('.')

import numpy as np
from models.config import build_cfg_from_argparse

from util.logger import Logger
import pickle
import datetime
from models.model import DLRwrapper

from util.train.data_reader import DataReader, DataReaderDialog

# Load config
cfg = build_cfg_from_argparse()


def load_train_data(max_num=0):
    imdb_file = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_VQA
    if cfg.dialog:
        data_reader = DataReaderDialog(
            imdb_file, shuffle=True, max_num=max_num,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            vocab_question_file=cfg.VOCAB_QUESTION_FILE,
            T_encoder=cfg.T_ENCODER,
            vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
            objects_feature_dir=cfg.OBJECTS_FEATURE_DIR,
            objects_max_num=cfg.W_FEAT,
            add_pos_enc=cfg.ADD_POS_ENC,
            pos_enc_dim=cfg.PE_DIM, pos_enc_scale=cfg.PE_SCALE)
    else:
        data_reader = DataReader(
            imdb_file, shuffle=True, max_num=max_num,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            vocab_question_file=cfg.VOCAB_QUESTION_FILE,
            T_encoder=cfg.T_ENCODER,
            vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
            objects_feature_dir=cfg.OBJECTS_FEATURE_DIR,
            objects_max_num=cfg.W_FEAT,
            add_pos_enc=cfg.ADD_POS_ENC,
            pos_enc_dim=cfg.PE_DIM, pos_enc_scale=cfg.PE_SCALE)
    num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
    num_choices = data_reader.batch_loader.answer_dict.num_vocab
    return data_reader, num_vocab, num_choices


def run_train_on_data(model, data_reader_train, run_eval=False,
                      data_reader_eval=None):
    model.train()
    infos = {}
    infos['best_val_acc'] = 0

    lr = cfg.TRAIN.SOLVER.LR
    loss_vqa_sum, loss_con_sum, loss_sub_sum = 0., 0., 0.
    correct, total, loss_sum, batch_num = 0, 0, 0., 0

    snapshot_dir = os.path.dirname(cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, 0))
    logger = Logger(os.path.join(snapshot_dir, 'log.txt'))

    train_s = datetime.datetime.now()

    for batch, n_sample, e in data_reader_train.batches(one_pass=False):

        n_epoch = cfg.TRAIN.START_EPOCH + e

        if cfg.dialog and cfg.loss_con and n_epoch >= cfg.loss_con_start_epoch:
            cfg.loss_con_start = True

        if n_sample == 0 and n_epoch > cfg.TRAIN.START_EPOCH:

            # save snapshot
            snapshot_file = cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, n_epoch)
            torch.save(model.state_dict(), snapshot_file)
            logger.write(train_log % tuple(train_res_log), True)
            train_e = datetime.datetime.now()
            train_time = train_e - train_s

            # run evaluation
            if run_eval:
                eval_res = run_eval_on_data(model, data_reader_eval, infos=infos, logger=logger)
                model.train()
                infos = eval_res['infos']

            eval_e = datetime.datetime.now()
            used_time = eval_e - train_s
            eval_time = eval_e - train_e

            logger.write('time: ' + str(used_time) + ' train: ' + str(train_time) + ' eval: ' + str(
                eval_time) + ' best acc = %.4f' % infos['best_val_acc'], True)
            train_s = eval_e

            # clear stats
            correct, total, loss_sum, batch_num = 0, 0, 0., 0
            loss_vqa_sum, loss_con_sum, loss_sub_sum = 0., 0., 0.

        if n_epoch >= cfg.TRAIN.MAX_EPOCH:
            break

        batch_res = model.run_batch(batch, train=True, lr=lr)
        correct += batch_res['num_correct']

        total += batch_res['batch_size']
        loss_sum += batch_res['loss'].item()
        batch_num += 1

        # log
        train_log = '\rTrain E %d S %d: loss=%.4f'
        train_res_log = [n_epoch + 1, total, loss_sum / batch_num]

        if cfg.dialog and batch_res['loss_con'] is not None:
            train_log += ', l_con=%.4f'
            loss_con_sum += batch_res['loss_con']
            train_res_log.append(loss_con_sum / batch_num)

        if cfg.loss_sub and batch_res['loss_sub'] is not None:
            train_log += ', l_sub=%.4f'
            loss_sub_sum += batch_res['loss_sub']
            train_res_log.append(loss_sub_sum / batch_num)

        if cfg.dialog and batch_res['loss_vqa'] is not None:
            train_log += ', l_vqa=%.4f'
            loss_vqa_sum += batch_res['loss_vqa']
            train_res_log.append(loss_vqa_sum / batch_num)

        train_log += ', acc=%.4f'
        train_res_log += [correct / total]
        train_log += ', lr=%.1e'
        train_res_log += [lr]

        print(train_log % tuple(train_res_log), end='')


def load_eval_data(max_num=0):
    imdb_file = cfg.IMDB_FILE % cfg.TEST.SPLIT_VQA
    if cfg.dialog_test:
        data_reader = DataReaderDialog(
            imdb_file, shuffle=False, max_num=max_num,
            batch_size=cfg.TEST.BATCH_SIZE,
            vocab_question_file=cfg.VOCAB_QUESTION_FILE,
            T_encoder=cfg.T_ENCODER,
            vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
            objects_feature_dir=cfg.OBJECTS_FEATURE_DIR,
            objects_max_num=cfg.W_FEAT,
            add_pos_enc=cfg.ADD_POS_ENC,
            pos_enc_dim=cfg.PE_DIM, pos_enc_scale=cfg.PE_SCALE)
    else:
        data_reader = DataReader(
            imdb_file, shuffle=False, max_num=max_num,
            batch_size=cfg.TEST.BATCH_SIZE,
            vocab_question_file=cfg.VOCAB_QUESTION_FILE,
            T_encoder=cfg.T_ENCODER,
            vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
            objects_feature_dir=cfg.OBJECTS_FEATURE_DIR,
            objects_max_num=cfg.W_FEAT,
            add_pos_enc=cfg.ADD_POS_ENC,
            pos_enc_dim=cfg.PE_DIM, pos_enc_scale=cfg.PE_SCALE)
    num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
    num_choices = data_reader.batch_loader.answer_dict.num_vocab
    return data_reader, num_vocab, num_choices


def run_eval_on_data(model, data_reader_eval, pred=False, infos=None, logger=None):
    model.eval()
    predictions = []
    answer_tokens = data_reader_eval.batch_loader.answer_dict.word_list
    loss_vqa_sum, loss_con_sum, loss_sub_sum = 0., 0., 0.
    correct, total, loss_sum, batch_num = 0, 0, 0., 0

    for batch, _, _ in data_reader_eval.batches(one_pass=True):
        batch_res = model.run_batch(batch, train=False)
        if pred:
            predictions.extend([
                {'questionId': q, 'prediction': answer_tokens[p]}
                for q, p in zip(batch['qid_list'], batch_res['predictions'])])

        correct += batch_res['num_correct']
        total += batch_res['batch_size']
        loss_sum += batch_res['loss'].item()
        batch_num += 1

        # log
        val_log = '\rEval S %d: loss=%.4f'
        val_res_log = [total, loss_sum / batch_num]

        if cfg.dialog_test and batch_res['loss_con'] is not None:
            val_log += ', l_con=%.4f'
            loss_con_sum += batch_res['loss_con']
            val_res_log.append(loss_con_sum / batch_num)

        if cfg.loss_sub and batch_res['loss_sub'] is not None:
            val_log += ', l_sub=%.4f'
            loss_sub_sum += batch_res['loss_sub']
            val_res_log.append(loss_sub_sum / batch_num)

        if cfg.dialog_test and batch_res['loss_vqa'] is not None:
            val_log += ', l_vqa=%.4f'
            loss_vqa_sum += batch_res['loss_vqa']
            val_res_log.append(loss_vqa_sum / batch_num)

        val_log += ', acc=%.4f'
        val_res_log += [correct / total]
        print(val_log % tuple(val_res_log), end='')

    if logger:
        logger.write(val_log % tuple(val_res_log), True)

    acc = correct / total
    if infos != None:
        if infos['best_val_acc'] < acc:
            infos['best_val_acc'] = acc
            snapshot_file = cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, 1234)
            snapshot_file = snapshot_file.replace('1234', 'best')
            torch.save(model.state_dict(), snapshot_file)
        else:
            acc = infos['best_val_acc']

    eval_res = {
        'correct': correct,
        'total': total,
        'accuracy': correct / total,
        'loss': loss_sum / batch_num,
        'predictions': predictions,
        'infos': infos}

    return eval_res


def dump_prediction_to_file(predictions, res_dir):
    pred_file = os.path.join(res_dir, 'pred_%s_%04d_%s.json' % (
        cfg.EXP_NAME, cfg.TEST.EPOCH, cfg.TEST.SPLIT_VQA))
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print('predictions written to %s' % pred_file)


def train():
    data_reader_train, num_vocab, num_choices = load_train_data()
    data_reader_eval, _, _ = load_eval_data(max_num=cfg.TRAIN.EVAL_MAX_NUM)

    # Load model
    model = DLRwrapper(num_vocab, num_choices)

    # Save snapshot
    snapshot_dir = os.path.dirname(cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, 0))
    os.makedirs(snapshot_dir, exist_ok=True)
    with open(os.path.join(snapshot_dir, 'cfg.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    if cfg.TRAIN.START_EPOCH > 0:
        print('resuming from epoch %d' % cfg.TRAIN.START_EPOCH)
        model.load_state_dict(torch.load(
            cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, cfg.TRAIN.START_EPOCH)))

    print('%s - train for %d epochs' % (cfg.EXP_NAME, cfg.TRAIN.MAX_EPOCH))
    run_train_on_data(
        model, data_reader_train, run_eval=cfg.TRAIN.RUN_EVAL,
        data_reader_eval=data_reader_eval)
    print('%s - train (done)' % cfg.EXP_NAME)


def test():
    data_reader_eval, num_vocab, num_choices = load_eval_data()

    # Load model
    model = DLRwrapper(num_vocab, num_choices)

    # Load test snapshot
    snapshot_file = cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, cfg.TEST.EPOCH)
    model_pt = torch.load(snapshot_file)
    model.load_state_dict(model_pt)

    res_dir = cfg.TEST.RESULT_DIR % (cfg.EXP_NAME, cfg.TEST.EPOCH)
    os.makedirs(res_dir, exist_ok=True)
    pred = cfg.TEST.DUMP_PRED

    if not pred:
        print('NOT writing predictions (set TEST.DUMP_PRED True to write)')

    if cfg.TEST.EPOCH == 'best':
        print('%s - test epoch %s' % (cfg.EXP_NAME, cfg.TEST.EPOCH))
    else:
        print('%s - test epoch %d' % (cfg.EXP_NAME, cfg.TEST.EPOCH))

    eval_res = run_eval_on_data(model, data_reader_eval, pred=pred)
    print(' ')
    if cfg.TEST.EPOCH == 'best':
        print('%s - test epoch %s: accuracy = %.4f' % (
            cfg.EXP_NAME, cfg.TEST.EPOCH, eval_res['accuracy']))
    else:
        print('%s - test epoch %d: accuracy = %.4f' % (
            cfg.EXP_NAME, cfg.TEST.EPOCH, eval_res['accuracy']))

    # write results
    if pred:
        dump_prediction_to_file(eval_res['predictions'], res_dir)

    eval_res.pop('predictions')

    if cfg.TEST.EPOCH == 'best':
        res_file = os.path.join(res_dir, 'res_%s_%s_%s.json' % (
            cfg.EXP_NAME, cfg.TEST.EPOCH, cfg.TEST.SPLIT_VQA))
    else:
        res_file = os.path.join(res_dir, 'res_%s_%04d_%s.json' % (
            cfg.EXP_NAME, cfg.TEST.EPOCH, cfg.TEST.SPLIT_VQA))

    with open(res_file, 'w') as f:
        json.dump(eval_res, f)


def seed_torch(seed=123):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup(cfg):

    # if cfg.subset != 'all':
    #     cfg.TRAIN.SPLIT_VQA = cfg.subset + '_' + cfg.TRAIN.SPLIT_VQA
    #     cfg.TEST.SPLIT_VQA = cfg.subset + '_' + cfg.TEST.SPLIT_VQA
    #     cfg.TRAIN.MAX_EPOCH = 10

    # cfg.dialog = True
    # cfg.TRAIN.SPLIT_VQA = 'train_dialog_balanced'
    
    if cfg.train:
        cfg.EXP_NAME = cfg.exp_id + '_' + cfg.EXP_NAME
    else:
        cfg.EXP_NAME = cfg.TEST.EVAL_ID + '_' + cfg.EXP_NAME

    seed_torch(cfg.seed)

    return cfg


if __name__ == '__main__':
    cfg = setup(cfg)
    if cfg.train:
        train()
    else:
        test()

