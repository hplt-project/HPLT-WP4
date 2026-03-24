import torch
import torch.nn.functional as F


class CollateFunctor:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, sentences):
        longest_source = max([sentence["subwords"].size(0) for sentence in sentences])
        numbers_of_words = [sentence["alignment"][-1] - 1 for sentence in sentences]
        longest_target = max(numbers_of_words)
        return {
            "subwords": torch.stack(
                [
                    F.pad(sentence["subwords"],
                    (0, longest_source - sentence["subwords"].size(0)), value=self.pad_index)
                    for sentence in sentences
                ]
            ),
            "alignment": torch.stack(
                [
                    F.pad(
                        F.one_hot(sentence["alignment"], num_classes=longest_target + 2).float(),
                        (0, 0, 0, longest_source - sentence["alignment"].size(0)), value=0.0)
                    for sentence in sentences
                ]
            ),
            "subword_lengths": torch.LongTensor([sentence["subwords"].size(0) for sentence in sentences]),
            "word_lengths": torch.LongTensor(numbers_of_words),
            "words": [sentence["words"] for sentence in sentences]
        }
