import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import dependency_decoding

from transformers import AutoModel


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class Classifier(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout):
        super().__init__()

        self.transform = nn.Sequential(
            nn.Linear(hidden_size, 2*2560),
            GEGLU(),
            nn.LayerNorm(2560, elementwise_affine=False),
            nn.Linear(2560, hidden_size, bias=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )
        self.initialize(hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.transform[0].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.transform[-1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.transform[0].bias.data.zero_()
        self.transform[-1].bias.data.zero_()

    def forward(self, x):
        return self.transform(x)


class ZeroClassifier(nn.Module):
    def forward(self, x):
        output = torch.zeros(x.size(0), x.size(1), 2, device=x.device, dtype=x.dtype)
        output[:, :, 0] = 1.0
        output[:, :, 1] = -1.0
        return output


class EdgeClassifier(nn.Module):
    def __init__(self, hidden_size, dep_hidden_size, vocab_size, dropout):
        super().__init__()

        self.head_dep_transform = nn.Sequential(
            nn.Linear(hidden_size, 2*2560),
            GEGLU(),
            nn.LayerNorm(2560, elementwise_affine=False),
            nn.Linear(2560, hidden_size, bias=False),
            nn.Dropout(dropout)
        )
        self.head_root_transform = nn.Sequential(
            nn.Linear(hidden_size, 2*2560),
            GEGLU(),
            nn.LayerNorm(2560, elementwise_affine=False),
            nn.Linear(2560, hidden_size, bias=False),
            nn.Dropout(dropout)
        )
        self.head_bilinear = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.head_linear_dep = nn.Linear(hidden_size, 1, bias=False)
        self.head_linear_root = nn.Linear(hidden_size, 1, bias=False)

        self.dep_dep_transform = nn.Sequential(
            nn.Linear(hidden_size, 2*2560),
            GEGLU(),
            nn.LayerNorm(2560, elementwise_affine=False),
            nn.Linear(2560, dep_hidden_size, bias=False),
            nn.Dropout(dropout)
        )
        self.dep_root_transform = nn.Sequential(
            nn.Linear(hidden_size, 2*2560),
            GEGLU(),
            nn.LayerNorm(2560, elementwise_affine=False),
            nn.Linear(2560, dep_hidden_size, bias=False),
            nn.Dropout(dropout)
        )
        self.dep_bilinear = nn.Parameter(torch.zeros(dep_hidden_size, dep_hidden_size, vocab_size))
        self.dep_linear_dep = nn.Linear(dep_hidden_size, vocab_size, bias=False)
        self.dep_linear_root = nn.Linear(dep_hidden_size, vocab_size, bias=False)
        self.dep_bias = nn.Parameter(torch.zeros(vocab_size))

        self.hidden_size = hidden_size
        self.dep_hidden_size = dep_hidden_size

        self.mask_value = float("-inf")
        self.initialize(hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.head_dep_transform[0].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.head_root_transform[0].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.dep_dep_transform[0].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.dep_root_transform[0].weight, mean=0.0, std=std, a=-2*std, b=2*std)

        nn.init.trunc_normal_(self.head_linear_dep.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.head_linear_root.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.dep_linear_dep.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.dep_linear_root.weight, mean=0.0, std=std, a=-2*std, b=2*std)

        self.head_dep_transform[0].bias.data.zero_()
        self.head_root_transform[0].bias.data.zero_()
        self.dep_dep_transform[0].bias.data.zero_()
        self.dep_root_transform[0].bias.data.zero_()

    def forward(self, head_x, dep_x, lengths, head_gold=None):
        head_dep = self.head_dep_transform(head_x[:, 1:, :])
        head_root = self.head_root_transform(head_x)
        head_prediction = torch.einsum("bkn,nm,blm->bkl", head_dep, self.head_bilinear, head_root / math.sqrt(self.hidden_size)) \
            + self.head_linear_dep(head_dep) + self.head_linear_root(head_root).transpose(1, 2)

        mask = (torch.arange(head_x.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)).unsqueeze(1).to(head_x.device)
        mask = mask | (torch.ones(head_x.size(1) - 1, head_x.size(1), dtype=torch.bool, device=head_x.device).tril(1) & torch.ones(head_x.size(1) - 1, head_x.size(1), dtype=torch.bool, device=head_x.device).triu(1))
        head_prediction = head_prediction.masked_fill(mask, self.mask_value)

        if head_gold is None:
            head_logp = head_prediction
            head_logp = F.pad(head_logp, (0, 0, 1, 0), value=torch.nan).cpu()
            head_gold = []
            for i, length in enumerate(lengths.tolist()):
                head = self.max_spanning_tree(head_logp[i, :length, :length])
                head = head + ((head_x.size(1) - 1) - len(head)) * [0]
                head_gold.append(torch.tensor(head))
            head_gold = torch.stack(head_gold).to(head_x.device)

        dep_dep = self.dep_dep_transform(dep_x[:, 1:])
        dep_root = dep_x.gather(1, head_gold.unsqueeze(-1).expand(-1, -1, dep_x.size(-1)).clamp(min=0))
        dep_root = self.dep_root_transform(dep_root)
        dep_prediction = torch.einsum("btm,mnl,btn->btl", dep_dep, self.dep_bilinear, dep_root / math.sqrt(self.dep_hidden_size)) \
            + self.dep_linear_dep(dep_dep) + self.dep_linear_root(dep_root) + self.dep_bias

        return head_prediction, dep_prediction, head_gold
    
    def max_spanning_tree(self, weight_matrix):
        weight_matrix = weight_matrix.clone()
        weight_matrix[weight_matrix == self.mask_value] = torch.nan
        # weight_matrix[:, 0] = torch.nan

        # we need to make sure that the root is the parent of a single node
        # first, we try to use the default weights, it should work in most cases
        parents, _ = dependency_decoding.chu_liu_edmonds(weight_matrix.numpy().astype(float))

        assert parents[0] == -1, f"{parents}\n{weight_matrix}"
        parents = parents[1:]

        # check if the root is the parent of a single node
        if parents.count(0) == 1:
            return parents
        
        # if not, we need to modify the weights and try all possibilities
        # we try to find the node that is the parent of the root
        best_score = float("-inf")
        best_parents = None

        for i in range(len(parents)):
            weight_matrix_mod = weight_matrix.clone()
            weight_matrix_mod[:i+1, 0] = torch.nan
            weight_matrix_mod[i+2:, 0] = torch.nan
            parents, score = dependency_decoding.chu_liu_edmonds(weight_matrix_mod.numpy().astype(float))
            parents = parents[1:]

            if score > best_score:
                best_score = score
                best_parents = parents

        def print_whole_matrix(matrix):
            for i in range(matrix.shape[0]):
                print(" ".join([str(x) for x in matrix[i]]))

        assert best_parents is not None, f"{best_parents}\n{print_whole_matrix(weight_matrix)}"
        return best_parents


class Model(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()

        self.bert = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
        self.n_layers = self.bert.config.num_hidden_layers
        args.hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.hidden_size, elementwise_affine=False)
        self.upos_layer_score = nn.Parameter(torch.zeros(self.n_layers + 1, dtype=torch.float))
        self.xpos_layer_score = nn.Parameter(torch.zeros(self.n_layers + 2, dtype=torch.float))
        self.feats_layer_score = nn.Parameter(torch.zeros(self.n_layers + 2, dtype=torch.float))
        self.lemma_layer_score = nn.Parameter(torch.zeros(self.n_layers + 2, dtype=torch.float))
        self.head_layer_score = nn.Parameter(torch.zeros(self.n_layers + 2, dtype=torch.float))
        self.dep_layer_score = nn.Parameter(torch.zeros(self.n_layers + 2, dtype=torch.float))

        self.upos_embedding = nn.Embedding(len(dataset.upos_vocab), args.hidden_size)
        std = math.sqrt(2.0 / (5.0 * args.hidden_size))
        nn.init.trunc_normal_(self.upos_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

        self.lemma_classifier = nn.ModuleDict({
            cls: Classifier(args.hidden_size, max(len(dataset.lemma_vocab[cls]) - 1, 1), args.dropout) if len(dataset.lemma_vocab[cls]) > 2 else ZeroClassifier()
            for cls in dataset.lemma_vocab.keys()
        })
        self.upos_classifier = Classifier(args.hidden_size, max(len(dataset.upos_vocab) - 1, 1), args.dropout) if len(dataset.upos_vocab) > 2 else ZeroClassifier()
        self.xpos_classifier = Classifier(args.hidden_size, max(len(dataset.xpos_vocab) - 1, 1), args.dropout) if len(dataset.xpos_vocab) > 2 else ZeroClassifier()
        self.feats_classifier = Classifier(args.hidden_size, max(len(dataset.feats_vocab) - 1, 1), args.dropout) if len(dataset.feats_vocab) > 2 else ZeroClassifier()
        self.edge_classifier = EdgeClassifier(args.hidden_size, 128, max(len(dataset.arc_dep_vocab) - 1, 1), args.dropout)
        self.aux_feats_classifiers = nn.ModuleDict({
            cls: Classifier(args.hidden_size, max(len(dataset.feats_classes_vocab[cls]) - 1, 1), args.dropout) if len(dataset.feats_classes_vocab[cls]) > 2 else ZeroClassifier()
            for cls in dataset.feats_classes_vocab
        })

    def forward(self, x, alignment_mask, subword_lengths, word_lengths, upos_gold=None, head_gold=None):
        padding_mask = (torch.arange(x.size(1)).unsqueeze(0) < subword_lengths.unsqueeze(1)).to(x.device)
        x = self.bert(x, padding_mask, output_hidden_states=True).hidden_states
        x = torch.stack(x, dim=0)
        x = torch.einsum("lbsd,bst->lbtd", x, alignment_mask) / alignment_mask.sum(1).unsqueeze(-1).unsqueeze(0).clamp(min=1.0)

        # upos_x = torch.einsum("lbtd, l -> btd", x, torch.softmax(self.upos_layer_score, dim=0))
        upos_x = (x[:, :, 1:-1, :] * torch.softmax(self.upos_layer_score, dim=0).view(-1, 1, 1, 1)).sum(0)
        upos_x = self.dropout(self.layer_norm(upos_x))
        upos_preds = self.upos_classifier(upos_x)

        if upos_gold is None:
            upos_gold = upos_preds.argmax(-1)

        upos_embedding = self.upos_embedding(upos_gold.clamp(min=0))
        upos_embedding = F.pad(upos_embedding, (0, 0, 1, 1), value=0.0)
        x = torch.cat([x, upos_embedding.unsqueeze(0)], dim=0)

        xpos_x = (x[:, :, 1:-1, :] * torch.softmax(self.xpos_layer_score, dim=0).view(-1, 1, 1, 1)).sum(0)
        feats_x = (x[:, :, 1:-1, :] * torch.softmax(self.feats_layer_score, dim=0).view(-1, 1, 1, 1)).sum(0)
        lemma_x = (x[:, :, 1:-1, :] * torch.softmax(self.lemma_layer_score, dim=0).view(-1, 1, 1, 1)).sum(0)
        head_x = (x[:, :, 0:-1, :] * torch.softmax(self.head_layer_score, dim=0).view(-1, 1, 1, 1)).sum(0)
        dep_x = (x[:, :, 0:-1, :] * torch.softmax(self.dep_layer_score, dim=0).view(-1, 1, 1, 1)).sum(0)

        xpos_x = self.dropout(self.layer_norm(xpos_x))
        feats_x = self.dropout(self.layer_norm(feats_x))
        lemma_x = self.dropout(self.layer_norm(lemma_x))
        head_x = self.dropout(self.layer_norm(head_x))
        dep_x = self.dropout(self.layer_norm(dep_x))

        lemma_preds = {
            cls: classifier(lemma_x)
            for cls, classifier in self.lemma_classifier.items()
        }
        xpos_preds = self.xpos_classifier(xpos_x)
        feats_preds = self.feats_classifier(feats_x)
        head_prediction, dep_prediction, head_liu = self.edge_classifier(head_x, dep_x, word_lengths, head_gold)
        aux_feats_prediction = {
            cls: classifier(feats_x)
            for cls, classifier in self.aux_feats_classifiers.items()
        }

        return lemma_preds, upos_preds, xpos_preds, feats_preds, aux_feats_prediction, head_prediction, dep_prediction, head_liu
