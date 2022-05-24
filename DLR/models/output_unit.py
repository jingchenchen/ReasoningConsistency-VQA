import torch
from torch import nn

from . import ops as ops
from .config import cfg

class Classifier(nn.Module):
    def __init__(self, num_choices):
        super().__init__()
        self.outQuestion = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        in_dim = 3 * cfg.CTX_DIM if cfg.OUT_QUESTION_MUL else 2 * cfg.CTX_DIM
        self.classifier_layer = nn.Sequential(
            nn.Dropout(1 - cfg.outputDropout),
            ops.Linear(in_dim, cfg.OUT_CLASSIFIER_DIM),
            nn.ELU(),
            nn.Dropout(1 - cfg.outputDropout),
            ops.Linear(cfg.OUT_CLASSIFIER_DIM, num_choices))

    def forward(self, x_att, vecQuestions):
        eQ = self.outQuestion(vecQuestions)
        if cfg.OUT_QUESTION_MUL:
            features = torch.cat([x_att, eQ, x_att*eQ], dim=-1)
        else:
            features = torch.cat([x_att, eQ], dim=-1)
        logits = self.classifier_layer(features)
        return logits
