from smart_open import open
import torch
from torch.utils.data import Dataset
from lemma_rule import gen_lemma_rule, apply_lemma_rule
from collections import Counter
from tqdm import tqdm
from collections import defaultdict


class Dataset(Dataset):
    def __init__(self, path: str, partition: str, tokenizer, forms_vocab=None, lemma_vocab=None, upos_vocab=None, xpos_vocab=None, feats_vocab=None, arc_dep_vocab=None, add_sep=True, random_mask=False, min_count=5):
        self.path = path
        full_texts, entries, current = [], [], []
        for line in open(path):
            if line.startswith("# text = "):
                full_texts.append(line[len("# text = "):].strip())
                continue
            elif line.startswith("#"):
                continue

            line = line.strip()

            if len(line) == 0:
                if len(current) == 0:
                    continue
                if len(current) > 512 - 2:
                    print(f"Skipping sentence of length {len(current)}")
                    current = []
                    continue
                entries.append(current)
                current = []
                continue

            res = [item.strip() for item in line.split("\t")]
            if not res[0].isdigit():
                continue
            current.append(res)

        self.entries = entries
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.n_embeddings = 1
        self.pad_index = self.tokenizer.pad_token_id
        self.add_sep = add_sep
        self.random_mask = random_mask

        self.forms = [[current[1] for current in entry] for entry in entries]

        self.subwords, self.alignment = [], []
        n_splits, n = 0, 0
        for i_sentence, sentence in enumerate(self.forms):
            subwords, alignment = [self.tokenizer.cls_token_id], [0]
            for i, word in enumerate(sentence):
                space_before = (i == 0) or (not "SpaceAfter=No" in entries[i_sentence][i - 1][-1])

                # very very ugly hack ;(
                offset = 2 if "<mask>" == self.tokenizer.mask_token else 1
                encoding = self.tokenizer(f"| {word}" if space_before else f"|{word}", add_special_tokens=False)
                subwords += encoding.input_ids[offset:]
                alignment += (len(encoding.input_ids) - offset) * [i + 1]

                # assert len(encoding.input_ids) > offset, f"{word} {encoding.input_ids}"
                # assert word == ''.join(tokenizer.decode(encoding.input_ids[offset:]).strip().split()), f"{word} != {tokenizer.decode(encoding.input_ids[offset:])}"

                if not word.isalpha():
                    continue
                n_splits += len(encoding.input_ids) - offset
                n += 1

            if self.add_sep:
                subwords.append(self.tokenizer.sep_token_id)
                alignment.append(alignment[-1] + 1)

            self.subwords.append(subwords)
            self.alignment.append(alignment)

        self.average_word_splits = n_splits / max(n, 1)
        print(self.average_word_splits)

        self.lemma = [[gen_lemma_rule(current[1], current[2], True) for current in entry] for entry in entries]
        self.upos = [[current[3] for current in entry] for entry in entries]
        self.xpos = [[current[4] for current in entry] for entry in entries]
        self.feats = [[current[5] for current in entry] for entry in entries]
        self.arc_head = [[int(current[6]) for current in entry] for entry in entries]
        self.arc_dep = [[current[7] for current in entry] for entry in entries]

        if forms_vocab:
            self.forms_vocab = forms_vocab
        else:
            self.forms_vocab = {item for sublist in self.forms for item in sublist}
            print(f"Size of forms vocabulary: {len(self.forms_vocab)}")

        if upos_vocab:
            self.upos_vocab = upos_vocab
        else:
            upos_counts = Counter([item for sublist in self.upos for item in sublist])
            self.upos_vocab = [i for i, j in upos_counts.items() if j >= min_count]
            self.upos_vocab = {i: item for i, item in enumerate(self.upos_vocab)}
            self.upos_vocab[-1] = '<unk>'
            print(f"Size of UPOS vocabulary: {len(self.upos_vocab)}")

        if xpos_vocab:
            self.xpos_vocab = xpos_vocab
        else:
            xpos_counts = Counter([item for sublist in self.xpos for item in sublist])
            self.xpos_vocab = [i for i, j in xpos_counts.items() if j >= min_count]
            self.xpos_vocab = {i: item for i, item in enumerate(self.xpos_vocab)}
            self.xpos_vocab[-1] = '<unk>'
            print(f"Size of XPOS vocabulary: {len(self.xpos_vocab)}")

        if feats_vocab:
            self.feats_vocab = feats_vocab
        else:
            feats_counts = Counter([item for sublist in self.feats for item in sublist])
            self.feats_vocab = [i for i, j in feats_counts.items() if j >= min_count]
            self.feats_vocab = {i: item for i, item in enumerate(self.feats_vocab)}
            self.feats_vocab[-1] = '<unk>'
            print(f"Size of FEATS vocabulary: {len(self.feats_vocab)}")

        if arc_dep_vocab:
            self.arc_dep_vocab = arc_dep_vocab
        else:
            arc_dep_counts = Counter([item for sublist in self.arc_dep for item in sublist])
            self.arc_dep_vocab = [i for i, j in arc_dep_counts.items() if j >= min_count]
            self.arc_dep_vocab = {i: item for i, item in enumerate(self.arc_dep_vocab)}
            self.arc_dep_vocab[-1] = '<unk>'
            print(f"Size of ARC_DEP vocabulary: {len(self.arc_dep_vocab)}")

        self.feats_classes_vocab = {
            feat.split("=")[0].strip()
            for feat_string in self.feats_vocab.values()
            for feat in feat_string.split("|")
            if feat != "_" and feat_string != "<unk>"
        }
        self.feats_classes_vocab = sorted(list(self.feats_classes_vocab))
        self.feats_classes_vocab = {cls: set() for cls in self.feats_classes_vocab}
        for feat_string in self.feats_vocab.values():
            if feat_string == "_" or feat_string == "<unk>":
                continue
            for feat in feat_string.split("|"):
                cls, value = feat.split("=")
                self.feats_classes_vocab[cls].add(value.strip())
        for cls in self.feats_classes_vocab.keys():
            self.feats_classes_vocab[cls] = sorted(list(self.feats_classes_vocab[cls]))
            self.feats_classes_vocab[cls] = {i: item for i, item in enumerate(self.feats_classes_vocab[cls])}
            self.feats_classes_vocab[cls][-1] = '<unk>'
        for cls in list(self.feats_classes_vocab.keys()):
            if len(self.feats_classes_vocab[cls]) <= 2:
                del self.feats_classes_vocab[cls]
                continue
        print(f"Size of FEATS classes: {len(self.feats_classes_vocab)}")
        print('\n'.join(self.feats_classes_vocab))
        for cls in self.feats_classes_vocab.keys():
            print(f"Size of {cls} UFeats: {len(self.feats_classes_vocab[cls])}")
            print('\t' + '\n\t'.join(self.feats_classes_vocab[cls].values()))

        if lemma_vocab:
            self.lemma_vocab = lemma_vocab
        else:
            lemma_rule_counts = Counter([
                (item["case"], item["prefix"], item["suffix"], item["absolute"])
                for sublist in self.lemma for item in sublist
            ])
            absolute_vocab = {item[3] for item in lemma_rule_counts.keys() if item[3].startswith("a")}

            form_lemma_set = [(token[1], token[2]) for entry in entries for token in entry]
            form_lemma_rules = {}

            for form, lemma in tqdm(form_lemma_set):
                base_rule = gen_lemma_rule(form, lemma, True)
                if base_rule["absolute"].startswith("a"):
                    form_lemma_rules[(form, lemma)] = {
                        "case": None, "prefix": None, "suffix": None, "absolute": base_rule["absolute"]
                    }
                    continue

                for (case, prefix, suffix, absolute), _ in lemma_rule_counts.most_common():
                    if absolute.startswith("a"):
                        continue
                    if apply_lemma_rule(form, {"case": case, "prefix": prefix, "suffix": suffix, "absolute": absolute}) == lemma:
                        form_lemma_rules[(form, lemma)] = {
                            "case": case, "prefix": prefix, "suffix": suffix, "absolute": absolute
                        }
                        break
                else:
                    form_lemma_rules[(form, lemma)] = {
                        "case": None, "prefix": None, "suffix": None, "absolute": base_rule["absolute"]
                    }

                if ("a" + lemma) in absolute_vocab:
                    form_lemma_rules[(form, lemma)]["absolute"] = "a" + lemma
                if form_lemma_rules[(form, lemma)]["case"] is None and lemma.islower():
                    form_lemma_rules[(form, lemma)]["case"] = "lower"

            rule_to_lemma_examples = {key: defaultdict(set) for key in ["case", "prefix", "suffix", "absolute"]}
            for (form, lemma), rule in form_lemma_rules.items():
                for rule_type in rule_to_lemma_examples.keys():
                    rule_to_lemma_examples[rule_type][rule[rule_type]].add(lemma)
            
            for (form, lemma), rule in form_lemma_rules.items():
                for rule_type in rule_to_lemma_examples.keys():
                    if len(rule_to_lemma_examples[rule_type][rule[rule_type]]) == 1:
                        form_lemma_rules[(form, lemma)][rule_type] = None
                        form_lemma_rules[(form, lemma)]["absolute"] = "a" + lemma
            
            self.lemma = [
                [
                    form_lemma_rules[(current[1], current[2])]
                    for current in entry
                ]
                for entry in entries
            ]

            lemma_counts = {
                rule_type: Counter([
                    item[rule_type] for sublist in self.lemma for item in sublist if item[rule_type] is not None
                ])
                for rule_type in rule_to_lemma_examples.keys()
            }
            self.lemma_vocab = {key: [i for i, j in counts.items() if j >= min_count] for key, counts in lemma_counts.items()}
            self.lemma_vocab = {key: {i: item for i, item in enumerate(vocab)} for key, vocab in self.lemma_vocab.items()}
            for key in self.lemma_vocab.keys():
                self.lemma_vocab[key][-1] = None
            print("Size of lemma vocabularies:")
            for key in self.lemma_vocab.keys():
                print(f"\t{key}: {len(self.lemma_vocab[key])}")

        self.lemma_indexer = {
            key: defaultdict(lambda: self.lemma_indexer[key][None], {item: i for i, item in self.lemma_vocab[key].items()})
            for key in self.lemma_vocab.keys()
        }
        self.upos_indexer = defaultdict(lambda: self.upos_indexer['<unk>'], {item: i for i, item in self.upos_vocab.items()})
        self.xpos_indexer = defaultdict(lambda: self.xpos_indexer['<unk>'], {item: i for i, item in self.xpos_vocab.items()})
        self.feats_indexer = defaultdict(lambda: self.feats_indexer['<unk>'], {item: i for i, item in self.feats_vocab.items()})
        self.arc_dep_indexer = defaultdict(lambda: self.arc_dep_indexer['<unk>'], {item: i for i, item in self.arc_dep_vocab.items()})
        self.feats_classes_indexer = {
            cls: defaultdict(lambda: -1, {item: i for i, item in self.feats_classes_vocab[cls].items()})
            for cls in self.feats_classes_vocab
        }

    # save state dict
    def state_dict(self):
        return {
            "forms_vocab": self.forms_vocab,
            "lemma_vocab": self.lemma_vocab,
            "upos_vocab": self.upos_vocab,
            "xpos_vocab": self.xpos_vocab,
            "feats_vocab": self.feats_vocab,
            "arc_dep_vocab": self.arc_dep_vocab,
            "feats_classes_vocab": self.feats_classes_vocab
        }

    # load state dict
    def load_state_dict(self, state_dict):
        self.forms_vocab = state_dict["forms_vocab"]
        self.lemma_vocab = state_dict["lemma_vocab"]
        self.upos_vocab = state_dict["upos_vocab"]
        self.xpos_vocab = state_dict["xpos_vocab"]
        self.feats_vocab = state_dict["feats_vocab"]
        self.arc_dep_vocab = state_dict["arc_dep_vocab"]
        self.feats_classes_vocab = state_dict["feats_classes_vocab"]

        #self.lemma_indexer = defaultdict(lambda: self.lemma_indexer['<unk>'], {item: i for i, item in self.lemma_vocab.items()})
        self.lemma_indexer = {
            key: defaultdict(lambda: self.lemma_vocab[key][None], {item: i for i, item in self.lemma_vocab[key].items()})
            for key in self.lemma_vocab
        }
        self.upos_indexer = defaultdict(lambda: self.upos_indexer['<unk>'], {item: i for i, item in self.upos_vocab.items()})
        self.xpos_indexer = defaultdict(lambda: self.xpos_indexer['<unk>'], {item: i for i, item in self.xpos_vocab.items()})
        self.feats_indexer = defaultdict(lambda: self.feats_indexer['<unk>'], {item: i for i, item in self.feats_vocab.items()})
        self.arc_dep_indexer = defaultdict(lambda: self.arc_dep_indexer['<unk>'], {item: i for i, item in self.arc_dep_vocab.items()})
        self.feats_classes_indexer = {
            cls: defaultdict(lambda: self.feats_classes_vocab[cls]['<unk>'], {item: i for i, item in self.feats_classes_vocab[cls].items()})
            for cls in self.feats_classes_vocab
        }

    def get_feats_classes(self, feats_string):
        if feats_string == "_" or feats_string == "<unk>":
            classes = {
                cls: -1
                for cls in self.feats_classes_vocab
            }
            return classes

        feats = {feat.split("=")[0]: feat.split("=")[1] for feat in feats_string.split("|")}
        classes = {
            cls: self.feats_classes_indexer[cls][feats[cls]] if cls in feats else -1
            for cls in self.feats_classes_vocab
        }
        return classes

    def __getitem__(self, index):
        subwords, alignment = [self.tokenizer.cls_token_id], [0]
        for i, word in enumerate(self.forms[index]):
            space_before = (i == 0) or (not "SpaceAfter=No" in self.entries[index][i - 1][-1])

            offset = 2 if "<mask>" == self.tokenizer.mask_token else 1
            encoding = self.tokenizer(f"| {word}" if space_before else f"|{word}", add_special_tokens=False)
            if self.random_mask and torch.rand([]).item() < 0.15:
                subwords += (len(encoding.input_ids) - offset) * [self.tokenizer.mask_token_id]
            else:
                subwords += encoding.input_ids[offset:]

            alignment += (len(encoding.input_ids) - offset) * [i + 1]

        if len(subwords) > 512 - 1:
            assert len(self.forms[index]) <= 512 - 2
            subwords, alignment = [self.tokenizer.cls_token_id], [0]
            for i, word in enumerate(self.forms[index]):
                space_before = (i == 0) or (not "SpaceAfter=No" in self.entries[index][i - 1][-1])

                offset = 2 if "<mask>" == self.tokenizer.mask_token else 1
                encoding = self.tokenizer(f"| {word}" if space_before else f"|{word}", add_special_tokens=False)
                if self.random_mask and torch.rand([]).item() < 0.15:
                    subwords += [self.tokenizer.mask_token_id]
                else:
                    subwords += encoding.input_ids[offset:offset+1]

                alignment += [i + 1]

        if self.add_sep:
            subwords.append(self.tokenizer.sep_token_id)
            alignment.append(alignment[-1] + 1)

        classes_per_word = [
            self.get_feats_classes(feats)
            for feats in self.feats[index]
        ]
        merged_classes = {
            cls: torch.LongTensor([classes_int[cls] for classes_int in classes_per_word])
            for cls in self.feats_classes_vocab
        }

        lemma_rules = {
            rule_type: torch.LongTensor(
                [
                    self.lemma_indexer[rule_type][lemma[rule_type]]
                    for lemma in self.lemma[index]
                ]
            )
            for rule_type in ["case", "prefix", "suffix", "absolute"]
        }

        return {
            "index": index,
            "subwords": torch.LongTensor(subwords),
            "alignment": torch.LongTensor(alignment),
            "is_unseen": torch.BoolTensor([form not in self.forms_vocab for form in self.forms[index]]),
            "lemma": lemma_rules,
            "upos": torch.LongTensor([self.upos_indexer[i] for i in self.upos[index]]),
            "xpos": torch.LongTensor([self.xpos_indexer[i] for i in self.xpos[index]]),
            "feats": torch.LongTensor([self.feats_indexer[i] for i in self.feats[index]]),
            "arc_head": torch.LongTensor(self.arc_head[index]),
            "arc_dep": torch.LongTensor([self.arc_dep_indexer[i] for i in self.arc_dep[index]]),
            "aux_feats_classes": merged_classes
        }

    def __len__(self):
        return len(self.forms)
