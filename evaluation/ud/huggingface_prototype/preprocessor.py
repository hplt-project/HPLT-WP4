import re
from string import punctuation

import torch

from collate_functor import CollateFunctor


class Preprocessor:
    def __init__(self, tokenizer):
        self.punct_pattern = re.compile(r"([{}])".format(re.escape(punctuation)))
        self.space_token = 'Ġ'
        self.collator = CollateFunctor(pad_index=0)
        self.tokenizer = tokenizer

    def preprocess(self, sentences):
        sentences = [re.sub(self.punct_pattern, r" \1 ", sentence).strip() for sentence in sentences]
        encoding = self.tokenizer(sentences)
        batch = []
        for encoding in encoding.encodings:
            batch.append({"subwords": encoding.ids, "tokens": encoding.tokens})
        for sentence in batch:
            alignment = [0]  # starting with cls token
            words = []
            current_alignment = 1
            current_word = []
            for token in sentence["tokens"][1:]:
                if (token == self.space_token) or (token == self.tokenizer.eos_token):
                    current_alignment += 1
                    words.append(self.tokenizer.convert_tokens_to_string(current_word))
                    current_word = []
                else:
                    current_word.append(token)
                alignment.append(current_alignment)
            sentence["alignment"] = torch.LongTensor(alignment)
            sentence["subwords"] = torch.LongTensor(sentence["subwords"])
            sentence["words"] = words
        return self.collator(batch)
