from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import ops as ops
from .config import cfg
from .graph_convolution import GC
from .output_unit import Classifier
from .input_unit import Encoder

def compute_consisitency_loss(logits,mask,answerIndices,num_choices):
    probs = F.softmax(logits, dim=-1)
    probs_one_hot = probs * F.one_hot(answerIndices, num_choices)
    probs_one_hot = probs_one_hot.sum(1)
    probs_diff = probs_one_hot.unsqueeze(0) - probs_one_hot.unsqueeze(1)
    probs_diff = - probs_diff * mask # only compute the difference of a quesiton and ts sub-question
    zero = torch.zeros_like(probs_diff)
    probs_diff = torch.where(probs_diff < 0, zero, probs_diff)
    loss_con = probs_diff.sum()
    return loss_con


def obtain_mask_for_consistency_loss(corrects):
    wrongs = (corrects == 0)
    correct_mask = corrects.unsqueeze(0) * corrects.unsqueeze(1)
    wrong_mask = wrongs.unsqueeze(0) * wrongs.unsqueeze(1)
    correct_wrong_mask = corrects.unsqueeze(0) * wrongs.unsqueeze(1)  # sub:correct ori: wrong
    wrong_correct_mask = wrongs.unsqueeze(0) * corrects.unsqueeze(1)  # ori:correct sub: wrong
    assert (wrongs + corrects).sum() == wrongs.shape[0]
    assert (correct_mask + wrong_mask + correct_wrong_mask + wrong_correct_mask).sum() == wrongs.shape[0] * \
           wrongs.shape[0]
    return correct_mask,wrong_mask,correct_wrong_mask,wrong_correct_mask



class SingleHop(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = ops.Linear(cfg.ENC_DIM, cfg.CTX_DIM)
        self.inter2att = ops.Linear(cfg.CTX_DIM, 1)

    def forward(self, kb, vecQuestions, imagesObjectNum):
        proj_q = self.proj_q(vecQuestions)
        interactions = F.normalize(kb * proj_q[:, None, :], dim=-1)
        raw_att = self.inter2att(interactions).squeeze(-1)
        raw_att = ops.apply_mask1d(raw_att, imagesObjectNum)
        att = F.softmax(raw_att, dim=-1)
        x_att = torch.bmm(att[:, None, :], kb).squeeze(1)
        return x_att

class DLRnet(nn.Module):
    def __init__(self, num_vocab, num_choices):
        super().__init__()
        if cfg.INIT_WRD_EMB_FROM_FILE:
            embeddingsInit = np.load(cfg.WRD_EMB_INIT_FILE)
            assert embeddingsInit.shape == (num_vocab - 1, cfg.WRD_EMB_DIM)
        else:
            embeddingsInit = np.random.randn(num_vocab - 1, cfg.WRD_EMB_DIM)
        self.encoder = Encoder(embeddingsInit)
        self.num_vocab = num_vocab
        self.num_choices = num_choices
        self.gc = GC()
        self.single_hop = SingleHop()
        self.classifier = Classifier(num_choices)

    def forward(self, batch,train=None):

        losses, loss_vqa,loss_con,loss_sub =[], None,None,None

        batchSize = len(batch['image_feat_batch'])
        questionIndices = torch.from_numpy(
            batch['input_seq_batch'].astype(np.int64)).cuda()
        questionLengths = torch.from_numpy(
            batch['seq_length_batch'].astype(np.int64)).cuda()
        answerIndices = torch.from_numpy(
            batch['answer_label_batch'].astype(np.int64)).cuda()
        images = torch.from_numpy(
            batch['image_feat_batch'].astype(np.float32)).cuda()
        imagesObjectNum = torch.from_numpy(
            np.sum(batch['image_valid_batch'].astype(np.int64), axis=1)).cuda()
        step = torch.from_numpy(
            batch['step_batch'].astype(np.int8)).cuda()

        if (cfg.dialog and train) or (cfg.dialog_test and not train):
            # the correspondences between sub-questions and compositional questions
            sub_mask = torch.from_numpy(
                batch['sub_mask_batch'].astype(np.int64)).cuda()
            ori_q = torch.from_numpy(
                batch['ori_q_batch'].astype(np.int8)).cuda()
            sub_q = ori_q.eq(0)
        else:
            sub_q = torch.zeros(batchSize).cuda()
            sub_mask = None

        # LSTM
        questionCntxWords, vecQuestions = self.encoder(
            questionIndices, questionLengths)

        # sub-questions in the reasoning process of the compositional question
        if sub_q.sum() > 0:
            # sub_mask = sub_mask * ori_q.unsqueeze(0)
            assert (sub_mask.sum(1) <= 1).sum() == sub_mask.sum(1).numel()
            idx_temp = torch.arange(0, sub_q.size(0)).cuda().long()
            sub_q_idx = idx_temp[sub_q]
            answerIndices_sub = answerIndices.index_select(0, sub_q_idx) # answers for sub-questions
            vecQuestions_sub = vecQuestions.index_select(0, sub_q_idx) # sub-questions
            imagesObjectNum_sub = imagesObjectNum.index_select(0, sub_q_idx) # objnums for sub-questions
            idx_temp = torch.max(sub_mask,1)[1]
            ori_idx = idx_temp.index_select(0, sub_q_idx)  # indices of ori-questions
            vecQuestions_ori_for_sub = vecQuestions.index_select(0, ori_idx) # ori-questions
            imagesObjectNum_ori = imagesObjectNum.index_select(0, ori_idx) # objnums for ori-questions
            assert imagesObjectNum_ori.equal(imagesObjectNum_sub) # check

        # graph convolution
        result = self.gc(
            images=images, q_encoding=vecQuestions,
            lstm_outputs=questionCntxWords, batch_size=batchSize,
            q_length=questionLengths, entity_num=imagesObjectNum,step=step,sub_mask=sub_mask,sub_q=sub_q)

        x_out = result['x_out']

        # Single-Hop
        x_att = self.single_hop(x_out, vecQuestions, imagesObjectNum)
        logits = self.classifier(x_att, vecQuestions)
        predictions, num_correct, corrects = self.add_pred_op(logits, answerIndices)
        loss = self.add_answer_loss_op(logits, answerIndices)
        losses.append(loss)

        # loss_sub
        if sub_q.sum() > 0:
            x_out_sub = result['x_out_sub']
            x_att_sub = self.single_hop(x_out_sub, vecQuestions_sub, imagesObjectNum_sub)
            logits_sub = self.classifier(x_att_sub, vecQuestions_sub)
            loss_sub = self.add_answer_loss_op(logits_sub, answerIndices_sub)
            loss_sub = cfg.loss_sub_w * loss_sub
            _, _, corrects_sub = self.add_pred_op(logits_sub, answerIndices_sub)
            losses.append(loss_sub)

        # consistency constraint
        if cfg.loss_con_start and sub_q.sum() > 0:

            correct_mask, _, _, wrong_correct_mask = obtain_mask_for_consistency_loss(corrects)
            valid_mask_con_ori = sub_mask * (correct_mask + wrong_correct_mask)
            loss_con_ori = compute_consisitency_loss(logits, valid_mask_con_ori, answerIndices,self.num_choices)
            loss_con_ori = cfg.loss_con_w * loss_con_ori

            ori_q = sub_q.eq(0)
            idx_temp = torch.arange(0, ori_q.size(0)).cuda().long()
            ori_q_idx = idx_temp[ori_q]
            logits_ori_q = logits.index_select(0, ori_q_idx)
            logits_con = torch.zeros_like(logits).cuda()
            logits_con.index_copy_(0, ori_q_idx, logits_ori_q)
            logits_con.index_copy_(0, sub_q_idx, logits_sub)

            corrects_step = torch.zeros_like(corrects).cuda()
            corrects_ori = corrects.index_select(0, ori_q_idx)
            corrects_step.index_copy_(0, ori_q_idx, corrects_ori)
            corrects_step.index_copy_(0, sub_q_idx, corrects_sub)
            correct_mask_step, _, _, wrong_correct_mask_step = obtain_mask_for_consistency_loss(corrects_step)
            valid_mask_con = sub_mask * (correct_mask_step + wrong_correct_mask_step)

            loss_con = compute_consisitency_loss(logits_con, valid_mask_con, answerIndices, self.num_choices)
            loss_con = cfg.loss_con_w * loss_con

            loss_con = loss_con + loss_con_ori
            losses.append(loss_con)


        if len(losses) > 1:
            loss_vqa = loss
            loss = sum(losses)

        return {"predictions": predictions,
                "batch_size": int(batchSize),
                "num_correct": int(num_correct),
                "sub_num": sub_q.sum(),
                "loss": loss,
                "loss_con": loss_con,
                "loss_vqa": loss_vqa,
                "loss_sub":loss_sub,
                "accuracy": float(num_correct * 1. / batchSize)}

    def add_pred_op(self, logits, answers):
        if cfg.MASK_PADUNK_IN_LOGITS:
            logits = logits.clone()
            logits[..., :2] += -1e30  # mask <pad> and <unk>
        preds = torch.argmax(logits, dim=-1).detach()
        corrects = (preds == answers)
        correctNum = torch.sum(corrects).item()
        preds = preds.cpu().numpy()
        return preds, correctNum, corrects

    def add_answer_loss_op(self, logits, answers, reduction_none=False):
        if reduction_none:
            assert cfg.TRAIN.LOSS_TYPE == "sigmoid"
            answerDist = F.one_hot(answers, self.num_choices).float()
            loss = F.binary_cross_entropy_with_logits(
                logits, answerDist, reduction='none').sum(1)
        else:
            if cfg.TRAIN.LOSS_TYPE == "softmax":
                loss = F.cross_entropy(logits, answers)
            elif cfg.TRAIN.LOSS_TYPE == "sigmoid":
                answerDist = F.one_hot(answers, self.num_choices).float()
                loss = F.binary_cross_entropy_with_logits(
                    logits, answerDist) * self.num_choices
            else:
                raise Exception("non-identified loss")
        return loss

class DLRwrapper():
    def __init__(self, num_vocab, num_choices):
        self.model = DLRnet(num_vocab, num_choices).cuda()
        self.trainable_params = [
            p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            self.trainable_params, lr=cfg.TRAIN.SOLVER.LR)
        self.lr = cfg.TRAIN.SOLVER.LR

        if cfg.USE_EMA:
            self.ema_param_dict = {
                name: p for name, p in self.model.named_parameters()
                if p.requires_grad}
            self.ema = ops.ExponentialMovingAverage(
                self.ema_param_dict, decay=cfg.EMA_DECAY_RATE)
            self.using_ema_params = False

    def train(self, training=True):
        self.model.train(training)
        if training:
            self.set_params_from_original()
        else:
            self.set_params_from_ema()

    def eval(self):
        self.train(False)

    def state_dict(self):
        # Generate state dict in training mode
        current_mode = self.model.training
        self.train(True)

        assert (not cfg.USE_EMA) or (not self.using_ema_params)
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict() if cfg.USE_EMA else None
        }

        # restore original mode
        self.train(current_mode)

    def load_state_dict(self, state_dict):
        # Load parameters in training mode
        current_mode = self.model.training
        self.train(True)

        assert (not cfg.USE_EMA) or (not self.using_ema_params)
        self.model.load_state_dict(state_dict['model'])

        if 'optimizer' in state_dict and cfg.train:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        else:
            print('Optimizer does not exist in checkpoint! '
                  'Loaded only model parameters.')

        if cfg.USE_EMA:
            if 'ema' in state_dict and state_dict['ema'] is not None:
                self.ema.load_state_dict(state_dict['ema'])
            else:
                print('cfg.USE_EMA is True, but EMA does not exist in '
                      'checkpoint! Using model params to initialize EMA.')
                self.ema.load_state_dict(
                    {k: p.data for k, p in self.ema_param_dict.items()})

        # restore original mode
        self.train(current_mode)

    def set_params_from_ema(self):
        if (not cfg.USE_EMA) or self.using_ema_params:
            return

        self.original_state_dict = deepcopy(self.model.state_dict())
        self.ema.set_params_from_ema(self.ema_param_dict)
        self.using_ema_params = True

    def set_params_from_original(self):
        if (not cfg.USE_EMA) or (not self.using_ema_params):
            return

        self.model.load_state_dict(self.original_state_dict)
        self.using_ema_params = False

    def run_batch(self, batch, train, lr=None):
        assert train == self.model.training
        assert (not train) or (lr is not None), 'lr must be set for training'

        if train:
            if lr != self.lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
            self.optimizer.zero_grad()
            batch_res = self.model.forward(batch,train)
            loss = batch_res['loss']
            loss.backward()
            if cfg.TRAIN.CLIP_GRADIENTS:
                nn.utils.clip_grad_norm_(
                    self.trainable_params, cfg.TRAIN.GRAD_MAX_NORM)
            self.optimizer.step()
            if cfg.USE_EMA:
                self.ema.step(self.ema_param_dict)
        else:
            with torch.no_grad():
                batch_res = self.model.forward(batch)

        return batch_res
