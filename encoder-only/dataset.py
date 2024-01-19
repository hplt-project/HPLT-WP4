import torch
import gzip


def apply_mask(args, input_ids, mask_ratios, replacement_ids, global_step):
    mask_p = args.mask_p_start + (args.mask_p_end - args.mask_p_start) * global_step / args.max_steps
    mask_p = max(mask_p, mask_ratios.min().item())

    mask = mask_ratios < mask_p
    target_ids = torch.where(mask, input_ids, -100)
    input_ids = torch.where(mask, replacement_ids, input_ids)

    real_mask_p = mask.sum().item() / mask_ratios.numel()

    return input_ids, target_ids, real_mask_p


class SpanMaskingStrategy:
    def __init__(self, n_special_tokens, random_p, keep_p, vocab_size, mask_token_id):
        self.n_special_tokens = n_special_tokens
        self.random_p = random_p
        self.keep_p = keep_p
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def __call__(self, tokens):
        mask_ratios = torch.full_like(tokens, fill_value=1.0, dtype=torch.float32)
        replacement_tokens = tokens.clone()

        n_to_mask = (tokens >= self.n_special_tokens).long().sum()
        n_masked = 0

        preservation_mask = tokens < self.n_special_tokens
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        counter = 256
        while n_to_mask > n_masked and counter > 0:
            span_length = torch.tensor([0]).geometric_(0.2) % 11
            span_length = span_length.clamp(1, 10).long().item()
            offset = torch.randint(-(span_length - 1), tokens.size(0) + span_length, []).item()
            sub_mask = torch.zeros_like(tokens, dtype=torch.bool)
            sub_mask[max(0, offset) : min(mask.size(0)-1, offset + span_length)] = True
            sub_mask[preservation_mask | mask] = False

            mask_ratios[sub_mask] = n_masked / n_to_mask

            random_number = torch.rand([]).item()
            if random_number < self.random_p:
                replacement_tokens[sub_mask] = torch.randint(
                    low=self.n_special_tokens,
                    high=self.vocab_size,
                    size=[sub_mask.sum().item()],
                    dtype=torch.long
                )
            elif random_number < (self.random_p + self.keep_p):
                replacement_tokens[sub_mask] = tokens[sub_mask]
            else:
                replacement_tokens[sub_mask] = self.mask_token_id

            mask |= sub_mask
            n_masked = mask.long().sum()
            counter -= 1

        return mask_ratios, replacement_tokens


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_file: str, tokenizer, args):
        self.path = input_file
        self.seq_length = args.seq_length
        self.short_p = args.short_p
        self.n_special_tokens = args.n_special_tokens

        self.mask_index = tokenizer.token_to_id("[MASK]")
        self.cls_index = tokenizer.token_to_id("[CLS]")
        self.sep_index = tokenizer.token_to_id("[SEP]")
        self.pad_index = tokenizer.token_to_id("[PAD]")

        self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

        with gzip.GzipFile(input_file, 'rb') as f:
            documents = torch.load(f)

        self.segments = [
            document[offset : offset + self.seq_length - 2]
            for document in documents
            for offset in range(0, len(document), (self.seq_length - 2) // 2)
            if len(document) > 0
        ]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens = self.segments[index]

        target_seq_length = self.seq_length - 2 if torch.rand([]).item() > self.short_p else torch.randint(1, self.seq_length - 2, []).item()
        tokens = tokens[:target_seq_length].long()

        while tokens.size(0) + 1 < target_seq_length:
            new_index = torch.randint(0, len(self.segments), []).item()
            new_tokens = self.segments[new_index].long()
            tokens = torch.cat([tokens, torch.LongTensor([self.sep_index]), new_tokens], dim=0)
            tokens = tokens[:target_seq_length]

        padding_length = (self.seq_length - 2) - tokens.size(0)
        segment = torch.cat([
            torch.LongTensor([self.cls_index]),
            tokens,
            torch.LongTensor([self.sep_index]),
            torch.LongTensor([self.pad_index] * padding_length)
        ])

        attention_mask = torch.cat([
            torch.zeros(len(tokens) + 2, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
        ])

        mask_ratios, replacement_tokens = self.masking_strategy(segment)

        return segment, attention_mask, mask_ratios, replacement_tokens

    def show_random_item(self):
        inputs, _, mask_ratios, replacement_tokens = self.__getitem__(self.randint(0, len(self)))
        print(' '.join(self.tokenizer.id_to_token(i) for i in inputs), flush=True)
        print(' '.join(self.tokenizer.id_to_token(i) for i in replacement_tokens), flush=True)
        print(mask_ratios, flush=True)


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, input_file: str, tokenizer, device_index, n_devices, args):
        self.path = input_file
        self.seq_length = 128
        self.n_special_tokens = args.n_special_tokens

        self.mask_index = tokenizer.token_to_id("[MASK]")
        self.cls_index = tokenizer.token_to_id("[CLS]")
        self.sep_index = tokenizer.token_to_id("[SEP]")
        self.pad_index = tokenizer.token_to_id("[PAD]")

        self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

        with gzip.GzipFile(input_file, 'rb') as f:
            documents = torch.load(f)

        self.segments = [
            document[offset : offset + self.seq_length - 2]
            for document in documents
            for offset in range(0, len(document), (self.seq_length - 2) // 2)
            if len(document) > 0
        ]
        self.segments = self.segments[:len(self.segments) // n_devices * n_devices]
        self.segments = self.segments[device_index::n_devices]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens = self.segments[index]

        target_seq_length = self.seq_length - 2
        tokens = tokens[:target_seq_length].long()

        padding_length = (self.seq_length - 2) - tokens.size(0)
        segment = torch.cat([
            torch.LongTensor([self.cls_index]),
            tokens,
            torch.LongTensor([self.sep_index]),
            torch.LongTensor([self.pad_index] * padding_length)
        ])

        attention_mask = torch.cat([
            torch.zeros(len(tokens) + 2, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
        ])

        mask_ratios, replacement_tokens = self.masking_strategy(segment)

        return segment, attention_mask, mask_ratios, replacement_tokens
