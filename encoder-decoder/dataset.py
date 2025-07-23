import gzip

import torch
import torch.nn.functional as F


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
        self.mask_indices = [self.tokenizer.token_to_id(f"[MASK]_{i}") for i in range(1,100)]

    def __call__(self, tokens):
        n_masked = torch.binomial((tokens >= self.n_special_tokens).float().sum(dim=0, keepdim=True), torch.FloatTensor([self.mask_p])).item()
        preservation_mask = tokens < self.n_special_tokens
        mask = torch.zeros_like(tokens, dtype=torch.bool)

        cycle_detection = 99
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

        input_tokens = torch.tensor(input_tokens, dtype=torch.long)
        output_tokens = torch.tensor(output_tokens, dtype=torch.long)

        return input_tokens, output_tokens
   

class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_file, tokenizer, seq_length=512, mask_p=0.15, short_p=0.1, random_p=0.1, keep_p=0.1):
        self.tokenizer = tokenizer

        self.seq_length = seq_length
        self.short_p = short_p
        self.n_special_tokens = 6

        self.masking_strategy = SpanMaskingStrategy(mask_p, tokenizer, self.n_special_tokens, padding_label_id=-100, random_p=random_p, keep_p=keep_p)

        self.cls_index = torch.tensor([self.tokenizer.token_to_id("[CLS]")], dtype=torch.long)
        self.sep_index = torch.tensor([self.tokenizer.token_to_id("[SEP]")], dtype=torch.long)
        self.pad_index = self.tokenizer.token_to_id("[PAD]")

        if not ".gz" in input_file:
            documents = torch.load(input_file)
        else:
            with gzip.GzipFile(input_file, 'rb') as f:
                documents = torch.load(f)
        try:
            assert len(documents) > 0
        except AssertionError:
            print(f"{input_file} is empty")
            raise AssertionError
        self.segments = [
            document[offset : offset + seq_length - 2]
            for document in documents
            for offset in range(0, len(document), (seq_length - 2) // 2)
            if len(document) > 0
        ]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        segment = self.segments[index]
        segment = torch.cat([self.cls_index, segment, self.sep_index]).long()

        inputs, outputs = self.masking_strategy(segment)
        return inputs, outputs


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

