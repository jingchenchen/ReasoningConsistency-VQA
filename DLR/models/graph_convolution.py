import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import ops as ops
from .config import cfg


class GC(nn.Module):
    def __init__(self):
        super().__init__()
        self.build_loc_ctx_init()
        self.build_extract_textual_command()
        self.build_graph_convolution()

    def build_loc_ctx_init(self):
        self.initKB = ops.Linear(cfg.D_FEAT, cfg.CTX_DIM)
        self.x_loc_drop = nn.Dropout(1 - cfg.stemDropout)
        self.initMem = nn.Parameter(torch.randn(1, 1, cfg.CTX_DIM))

    def build_extract_textual_command(self):
        self.qInput = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
        for t in range(cfg.ITER_NUM):
            qInput_layer2 = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
            setattr(self, "qInput%d" % t, qInput_layer2)
        self.cmd_inter2logits = ops.Linear(cfg.CMD_DIM, 1)

    def build_graph_convolution(self):
        self.read_drop = nn.Dropout(1 - cfg.readDropout)
        self.project_x_loc = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.project_x_ctx = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        self.queries = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.keys = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.vals = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        self.proj_queries = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        self.mem_update = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)
        self.combine_kb = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)

    def forward(self, images, q_encoding, lstm_outputs, batch_size, q_length,
                entity_num, step=None, sub_mask=None, sub_q=None):
        x_ctx_all = []

        # local representations and contextual representations
        x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(images)

        if sub_q.sum() > 0:
            idx_temp = torch.arange(0, sub_q.size(0)).cuda().long()
            sub_q_idx = idx_temp[sub_q]
            # iter of graph convolution convolutions for sub-questions
            step_sub_q = step.index_select(0, sub_q_idx)
            # iteration indeices for sub-questions
            iter_idx_sub_q = step_sub_q.sum(1) - 1
            # contextual representations of sub-questions in the reasoning process of a compositional question
            x_ctx_sub = torch.zeros(sub_q.sum(), x_ctx.shape[1], x_ctx.shape[2]).cuda()
            # index of the original question
            q_idx = torch.max(sub_mask,1)[1]
            # index of the original question for each sub-question
            ori_q_idx = q_idx.index_select(0, sub_q_idx)

        for t in range(cfg.ITER_NUM):

            x_ctx = self.run_graph_convolution_iter(
                q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
                x_ctx_var_drop, entity_num, t)

            x_ctx_all.append(x_ctx)
            if sub_q.sum() > 0:
                sub_q_curr = (iter_idx_sub_q == t)
                if sub_q_curr.sum() > 0:
                    # the index of a sub-question need to be answered at t-th iteration
                    idx_temp = torch.arange(0, sub_q_curr.size(0)).cuda().long()
                    sub_q_idx_curr = idx_temp[sub_q_curr]
                    # the index of the compositional question for the sub-question
                    ori_q_idx_curr = ori_q_idx.index_select(0, sub_q_idx_curr)
                    x_ctx_curr = x_ctx.index_select(0, ori_q_idx_curr)
                    x_ctx_sub.index_copy_(0,sub_q_idx_curr,x_ctx_curr)

        # output for a question
        x_ctx_all = torch.stack(x_ctx_all).transpose(0,1)
        idx = (step.sum(1) - 1).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(x_ctx.shape[0],1,x_ctx.shape[1],x_ctx.shape[2])
        x_ctx = x_ctx_all.gather(1,idx).squeeze(1)
        x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1))

        if sub_q.sum() > 0:
            # output for a sub-question in the reasoning process of a compositional question
            assert ((x_ctx_sub.sum(2).sum(1)) != 0).sum() == sub_q.sum()
            x_loc_sub = x_loc.index_select(0, ori_q_idx)
            x_out_sub = self.combine_kb(torch.cat([x_loc_sub, x_ctx_sub], dim=-1))
        else:
            x_out_sub = None

        result = {
            "x_out": x_out,
            "x_out_sub": x_out_sub
        }

        return result

    def extract_textual_command(self, q_encoding, lstm_outputs, q_length, t):
        qInput_layer2 = getattr(self, "qInput%d" % t)
        act_fun = ops.activations[cfg.CMD_INPUT_ACT]
        q_cmd = qInput_layer2(act_fun(self.qInput(q_encoding)))
        raw_att = self.cmd_inter2logits(
            q_cmd[:, None, :] * lstm_outputs).squeeze(-1)
        raw_att = ops.apply_mask1d(raw_att, q_length)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
        return cmd

    def graph_convolution(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num):
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)
        queries = self.queries(x_joint) * self.proj_queries(cmd)[:, None, :]
        keys = self.keys(x_joint)
        vals = self.vals(x_joint)
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(cfg.CTX_DIM))
        edge_score = ops.apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        x_ctx_tmp = torch.bmm(edge_prob, vals)
        x_ctx_new = self.mem_update(torch.cat([x_ctx, x_ctx_tmp], dim=-1))
        return x_ctx_new

    def run_graph_convolution_iter(
            self, q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
            x_ctx_var_drop, entity_num, t):
        # command vector
        cmd = self.extract_textual_command(
                q_encoding, lstm_outputs, q_length, t)
        # contextual representation
        x_ctx = self.graph_convolution(
            cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num)
        return x_ctx

    def loc_ctx_init(self, images):
        images = F.normalize(images, dim=-1)
        x_loc = self.initKB(images)
        x_loc = self.x_loc_drop(x_loc)
        x_ctx = self.initMem.expand(x_loc.size())
        x_ctx_var_drop = ops.generate_scaled_var_drop_mask(
            x_ctx.size(),
            keep_prob=(cfg.memoryDropout if self.training else 1.))
        return x_loc, x_ctx, x_ctx_var_drop
