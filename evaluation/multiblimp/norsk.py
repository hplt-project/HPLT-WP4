from difflib import SequenceMatcher
from functools import reduce


def mask(x1, x2, mask_symbol='▁<extra_id_$>', eos='</s>'):
    x1 += eos
    x2 += eos
    s = SequenceMatcher(None, x1, x2)
    source_1, source_2, target_1, target_2 = '', '', '', ''
    last_match = None
    matching_blocks = s.get_matching_blocks()
    matching_blocks = [block for block in matching_blocks if block.size > 0]

    source_mask_counter = 0
    target_mask_counter = 0
    for i, match in enumerate(matching_blocks):

        is_first_match = i == 0
        if is_first_match:
            source_1 += x1[match.a: match.a + match.size]
            source_2 += x2[match.b: match.b + match.size]
            if (match.a == 0) and (match.b == 0): # not sentence beginning
                current_mask = mask_symbol.replace('$', str(target_mask_counter))
                target_1 += current_mask
                target_2 += current_mask
                target_mask_counter += 1
            else:
                current_mask = mask_symbol.replace('$', str(source_mask_counter))
                source_1 = current_mask + source_1
                source_2 = current_mask + source_2
                current_mask = mask_symbol.replace('$', str(target_mask_counter))
                target_1 += x1[:match.a] + current_mask
                target_2 += x2[:match.b] + current_mask
                source_mask_counter += 1
                target_mask_counter += 1
        else:
            current_mask = mask_symbol.replace('$', str(source_mask_counter))
            source_1 += current_mask + x1[match.a: match.a + match.size]
            source_2 += current_mask + x2[match.b: match.b + match.size]
            source_mask_counter += 1

            current_mask = mask_symbol.replace('$', str(target_mask_counter))
            target_1 += x1[last_match.a + last_match.size:match.a] + current_mask
            target_2 += x2[last_match.b + last_match.size:match.b] + current_mask
            target_mask_counter += 1

        last_match = match
    return source_1.strip(), source_2.strip(), target_1.strip(), target_2.strip()


def score_norsk(dataset, ilm_model, mask_symbol, eos, max_length=None):
    def score_norsk_pair(row):
        sen_len = len(ilm_model.tokenizer.tokenize(row["source"]))
        wrong_sen_len = len(ilm_model.tokenizer.tokenize(row["correction"]))

        if (max_length is not None) and (
                (sen_len >= max_length) or (wrong_sen_len >= max_length)
        ):
            return 0.0, 0.0
        wrong_sen, sen, wrong_target, target = mask(row["source"], row["correction"], mask_symbol, eos)
        stimuli = [sen, wrong_sen]
        answers = [target, wrong_target]
        sen_prob, wrong_prob = ilm_model.conditional_score(
            prefix=stimuli,
            stimuli=answers,
            reduction=lambda x: x
        )
        sen_nll = -sen_prob.sum().item()
        wrong_nll = -wrong_prob.sum().item()
        delta = wrong_nll - sen_nll
        delta_gt_zero = int(delta > 0)
        return {
            "sen_prob": sen_prob.tolist(),
            "wrong_prob": wrong_prob.tolist(),
            "sen_nll": sen_nll,
            "wrong_nll": wrong_nll,
            "delta": delta,
            "delta_gt_zero": delta_gt_zero,
        }

    dataset = dataset.map(score_norsk_pair, batched=False)
    print(f'acc {reduce(lambda x, y: x+y, dataset["delta_gt_zero"]) / dataset.num_rows}')
