import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle
import gzip


class Indexer:
    def __init__(self, documents):
        lengths = [len(document) for document in documents]
        self.cumsum = torch.LongTensor([0] + lengths).cumsum(dim=0)

    def get_indices(self, index):
        document_index = torch.searchsorted(self.cumsum, index, right=True).item() - 1
        segment_index = index - self.cumsum[document_index]
        return document_index, segment_index

    def __len__(self):
        return self.cumsum[-1].item()


class CollateFunctor:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, sentences: list):
        source_ids, target_ids = zip(*sentences)
        source_ids, source_mask = self.collate_sentences(source_ids, self.pad_id)
        target_ids, _ = self.collate_sentences(target_ids, self.pad_id)
        return source_ids, source_mask, target_ids

    def collate_sentences(self, sentences: list, pad_id):
        lengths = [sentence.size(0) for sentence in sentences]
        max_length = max(lengths)

        subword_ids = torch.stack([
            F.pad(sentence, (0, max_length - length), value=pad_id)
            for length, sentence in zip(lengths, sentences)
        ])
        attention_mask = subword_ids == pad_id

        return subword_ids, attention_mask


class SpanMaskingStrategy:
    def __init__(self, mask_p, tokenizer, n_special_tokens, padding_label_id=-100, random_p=0.1, keep_p=0.1):
        self.mask_p = mask_p
        self.random_p = random_p
        self.keep_p = keep_p
        self.tokenizer = tokenizer
        self.n_special_tokens = n_special_tokens
        self.padding_label_id = padding_label_id
        self.mask_indices = [self.tokenizer.token_to_id(f"[MASK_{i}]") for i in range(100)]

    def __call__(self, tokens):
        n_masked = torch.binomial((tokens >= self.n_special_tokens).float().sum(dim=0, keepdim=True), torch.FloatTensor([self.mask_p])).item()
        preservation_mask = tokens < self.n_special_tokens
        mask = torch.zeros_like(tokens, dtype=torch.bool)

        cycle_detection = 100
        while mask.sum() <= n_masked and cycle_detection > 0:
            span_length = torch.tensor([0]).geometric_(1/3).item()
            offset = torch.randint(-(span_length - 1), tokens.size(0) + span_length, []).item()
            mask[max(0, offset) : min(mask.size(0)-1, offset + span_length)] = True
            mask[preservation_mask] = False

            cycle_detection -= 1

        input_tokens, output_tokens = [], []
        span_i = 0
        for i in range(mask.size(0)):
            if mask[i] and (i == 0 or not mask[i - 1]):
                input_tokens.append(self.mask_indices[span_i])
                output_tokens.append(self.mask_indices[span_i])
                output_tokens.append(tokens[i].item())
                span_i += 1
            elif mask[i]:
                output_tokens.append(tokens[i].item())
            else:
                input_tokens.append(tokens[i].item())

        output_tokens.append(self.mask_indices[span_i])

        input_tokens = torch.tensor(input_tokens)
        output_tokens = torch.tensor(output_tokens)

        return input_tokens, output_tokens
   

class Dataset(Dataset):
    def __init__(self, input_file, tokenizer, seq_length=512, mask_p=0.15, short_p=0.1, random_p=0.1, keep_p=0.1):
        self.tokenizer = tokenizer

        self.seq_length = seq_length
        self.short_p = short_p
        self.n_special_tokens = 6

        self.masking_strategy = SpanMaskingStrategy(mask_p, tokenizer, self.n_special_tokens, padding_label_id=-100, random_p=random_p, keep_p=keep_p)

        self.cls_index = torch.tensor([self.tokenizer.token_to_id("[CLS]")], dtype=torch.long)
        self.sep_index = torch.tensor([self.tokenizer.token_to_id("[SEP]")], dtype=torch.long)
        self.pad_index = self.tokenizer.token_to_id("[PAD]")

        # every document contains a list of sentences, which are themselves np arrays of integers
        with gzip.open(input_file, "rb") as f:
            self.documents = pickle.load(f)

        self.documents = [
            [torch.from_numpy(sentence) for sentence in document]
            for document in self.documents
        ]
        self.documents = [document for document in self.documents if len(document) > 0]
        self.indexer = Indexer(self.documents)

    def __len__(self):
        return len(self.indexer)

    def rand(self):
        return torch.rand(1).item()

    def randint(self, low, high):
        return torch.randint(low=low, high=high, size=(1,)).item()

    def __getitem__(self, index):
        tokens = self.get_segment(index)
        inputs, outputs = self.masking_strategy(tokens)
        return inputs, outputs

    def get_segment(self, index):
        document_index, sentence_index = self.indexer.get_indices(index)

        document = self.documents[document_index]
        sentence = document[sentence_index]

        if sentence.size(0) == 0:
            segment = []
        else:
            segment = [document[sentence_index]]

        total_length = sentence.size(0)
        sentence_index += 1

        while total_length <= self.seq_length - 2 and sentence_index < len(document):
            sentence = document[sentence_index]

            if sentence.size(0) == 0:
                sentence_index += 1
                continue

            segment.append(sentence)
            total_length += sentence.size(0)
            sentence_index += 1

        tokens = torch.cat(segment)

        target_seq_length = self.seq_length - 2 if self.rand() > self.short_p else self.randint(1, self.seq_length - 2)
        tokens = tokens[:target_seq_length]
        segment = torch.cat([self.cls_index, tokens, self.sep_index]).long()
        return segment

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            return tokens_a, tokens_b
        cut = total_length - max_num_tokens
        cut_left = self.randint(max(0, cut - len(tokens_b) + 1), min(cut + 1, len(tokens_a)))
        cut_right = cut - cut_left
        tokens_a = tokens_a[:len(tokens_a) - cut_left]
        tokens_b = tokens_b[:len(tokens_b) - cut_right]
        return tokens_a, tokens_b


if __name__ == "__main__":
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file("wordpiece.json")
    masking = SpanMaskingStrategy(0.15, tokenizer, 108, padding_label_id=-100, random_p=0.1, keep_p=0.1)
    s = """Maskinlæring (ML) er i dag på alles lepper. ML muliggjør løsninger og tjenester som vi for noen år siden ikke kunne forestille oss. ML er også på full fart inn i offentlig sektor. I hvilken grad kan vi stole rådene ML-tjenestene gir? Hvordan kan vi gjøre ML mer fortålig og hva er de etiske utfordringene forbundet med ML?"""
    inputs, outputs = masking(torch.tensor(tokenizer.encode(s).ids))
    print(tokenizer.decode(inputs.tolist(), skip_special_tokens=False))
    print(tokenizer.decode(outputs.tolist(), skip_special_tokens=False))
    exit()

    dataset = MlmDataset("data/pretrain/tokenized/train_0.pickle.gz", tokenizer, "whole_word")
    
    inputs, attention_mask, outputs = dataset[21000]

    for i, output in enumerate(outputs.tolist()):
        if output != -100:
            print(i, output, tokenizer.id_to_token(output))

