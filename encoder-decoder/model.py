import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _softmax_backward_data as _softmax_backward_data


class T5(nn.Module):
    def __init__(self, config, pad_id=None):
        super().__init__()
        self.embedding = WordEmbedding(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.classifier = Classifier(config, self.embedding.word_embedding.weight)
        self.pad_id = pad_id

    def get_contextualized(self, input_ids, attention_mask):
        static_embeddings = self.embedding(input_ids)
        contextualized_embeddings = self.encoder(static_embeddings, attention_mask.unsqueeze(1).unsqueeze(2))
        return contextualized_embeddings

    def forward(self, source_ids, target_ids, attention_mask):
        contextualized_embeddings = self.get_contextualized(source_ids, attention_mask)[-1]
        target_embeddings = self.decoder(
            self.embedding(target_ids[:-1, :]),
            contextualized_embeddings,
            attention_mask.unsqueeze(1).unsqueeze(2)
        )

        gold_labels = target_ids[1:, :]
        target_embeddings = torch.index_select(target_embeddings.flatten(0, 1), 0, torch.nonzero(gold_labels.flatten() != self.pad_id).squeeze())

        prediction = self.classifier(target_embeddings)

        gold_labels = gold_labels.flatten()
        gold_labels = gold_labels[gold_labels != self.pad_id]

        loss = F.cross_entropy(prediction, gold_labels)

        with torch.no_grad():
            accuracy = (prediction.argmax(-1) == gold_labels).float().mean()

        return loss, accuracy


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_relative_embedding = RelativeEmbedding(config)
        self.cross_relative_embedding = RelativeEmbedding(config)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        for i, layer in enumerate(self.layers):
            layer.mlp.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

    def forward(self, x, encoder_output, encoder_padding_mask):
        self_relative_embedding = self.self_relative_embedding()
        cross_relative_embedding = self.cross_relative_embedding()

        autoreg_mask = torch.triu(
            torch.full((x.size(0), x.size(0)), True, device=x.device),
            diagonal=1
        )

        for layer in self.layers:
            x = layer(x, autoreg_mask, encoder_output, encoder_padding_mask, self_relative_embedding, cross_relative_embedding)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.relative_embedding = RelativeEmbedding(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

        for i, layer in enumerate(self.layers):
            layer.mlp.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
    
    def forward(self, hidden_states, attention_mask):
        relative_embedding = self.relative_embedding()
        hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states.append(
                layer(hidden_states[-1], attention_mask, relative_embedding)
            )

        return hidden_states


class Classifier(nn.Module):
    def __init__(self, config, subword_embedding):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
#            nn.Linear(config.hidden_size, config.hidden_size),
#            nn.GELU(),
#            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(subword_embedding.size(1), subword_embedding.size(0))
        )
        self.initialize(config.hidden_size, subword_embedding)

    def initialize(self, hidden_size, embedding):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
#        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.nonlinearity[-1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
#        self.nonlinearity[-1].weight = embedding
#        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x):
        x = self.nonlinearity(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = Attention(config)
        self.cross_attention = Attention(config)
        self.mlp = FeedForward(config)

    def forward(self, x, autoreg_mask, encoder_output, encoder_padding_mask, self_relative_embedding, cross_relative_embedding):
        x = x + self.self_attention(x, x, autoreg_mask, self_relative_embedding)
        x = x + self.cross_attention(x, encoder_output, encoder_padding_mask, cross_relative_embedding)
        x = x + self.mlp(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = FeedForward(config)

    def forward(self, x, padding_mask, relative_embedding):
        x = x + self.attention(x, x, padding_mask, relative_embedding)
        x = x + self.mlp(x)
        return x


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate, approximate='tanh')
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return self.mlp(x)


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self, x, mask, dim):
        self.dim = dim
        x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, self.dim)
        x.masked_fill_(mask, 0.0)
        self.save_for_backward(x)
        return x

    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output.dtype)
        return inputGrad, None, None


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.in_proj_q = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.in_proj_k = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.in_proj_v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        position_indices = torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(1) \
            - torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0)
        position_indices = self.make_log_bucket_position(position_indices, config.position_bucket_size, config.max_position_embeddings)
        position_indices = config.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)
        self.initialize()

    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos))
        log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position-1) / mid) * (mid - 1)).int() + mid
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        return bucket_pos

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.in_proj_q.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_k.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_v.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.in_proj_q.bias.data.zero_()
        self.in_proj_k.bias.data.zero_()
        self.in_proj_v.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def compute_attention_scores(self, q, kv, relative_embedding):
        key_len, batch_size, _ = kv.shape
        query_len, _, _ = q.shape

        q = self.pre_layer_norm(q)
        kv = self.pre_layer_norm(kv)

        query = self.in_proj_q(q)  # shape: [T, B, D]
        key = self.in_proj_k(kv)  # shape: [T, B, D]
        value = self.in_proj_v(kv)  # shape: [T, B, D]

        query_pos = self.in_proj_q(self.dropout(relative_embedding))  # shape: [2T-1, 2D]
        query_pos = F.embedding(self.position_indices[:query_len, :key_len], query_pos)  # shape: [T, T, 2D]
        query_pos = query_pos.view(query_len, key_len, self.num_heads, self.head_size)

        key_pos = self.in_proj_k(self.dropout(relative_embedding))  # shape: [2T-1, 2D]
        key_pos = F.embedding(self.position_indices[:query_len, :key_len], key_pos)  # shape: [T, T, 2D]
        key_pos = key_pos.view(query_len, key_len, self.num_heads, self.head_size)

        query = query.reshape(query_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        key = key.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        value = value.view(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)

        query = query.view(batch_size, self.num_heads, query_len, self.head_size)
        key = key.view(batch_size, self.num_heads, key_len, self.head_size)
        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        attention_scores.add_(torch.einsum("bhqd,qkhd->bhqk", query, key_pos * self.scale))
        attention_scores.add_(torch.einsum("bhkd,qkhd->bhqk", key * self.scale, query_pos))

        return attention_scores, value

    def compute_output(self, attention_probs, value):
        attention_probs = self.dropout(attention_probs)
        context = torch.bmm(attention_probs.flatten(0, 1), value)  # shape: [B*H, Q, D]
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.hidden_size)  # shape: [Q, B, H*D]
        context = self.out_proj(context)
        context = self.post_layer_norm(context)
        context = self.dropout(context)
        return context

    def forward(self, q, kv, attention_mask, relative_embedding):
        attention_scores, value = self.compute_attention_scores(q, kv, relative_embedding)
        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)
        return self.compute_output(attention_probs, value)


class WordEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids):
        word_embedding = self.dropout(self.word_layer_norm(self.word_embedding(input_ids)))
        return word_embedding


class RelativeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.hidden_size))
        self.relative_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self):
        relative_embeddings = self.relative_layer_norm(self.relative_embedding)
        return relative_embeddings
