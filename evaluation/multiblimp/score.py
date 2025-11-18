import pandas as pd
import swifter


def run_score_pair(df_row_data, model, mask_1, mask_2, is_encoder):
    sen_prob, wrong_prob = score_pair(
            model, df_row_data.sen, df_row_data.wrong_sen, is_encoder, df_row_data['head'], df_row_data.swap_head, mask_1, mask_2,
        )

    sen_nll = -sen_prob.sum().item()
    wrong_nll = -wrong_prob.sum().item()
    return {
        "sen_prob": sen_prob.tolist(),
        "wrong_prob": wrong_prob.tolist(),
        "sen_nll": sen_nll,
        "wrong_nll": wrong_nll,
        }


def score_tse(model, fn: str, is_encoder: bool, mask_1: str, mask_2: str):
    tse_df = pd.read_csv(fn, sep="\t")

    tse_df["sen_prob"] = pd.Series(dtype=object).astype(object)
    tse_df["wrong_prob"] = pd.Series(dtype=object).astype(object)

    print(tse_df.shape, flush=True)
    tse_df = tse_df.swifter.apply(run_score_pair, axis=1, result_type='expand', args=(model, mask_1, mask_2, is_encoder))
    tse_df["delta"] = tse_df.wrong_nll - tse_df.sen_nll
    return tse_df


def score_pair(ilm_model, sen, wrong_sen, is_encoder, head, swap_head, mask_1, mask_2):
    stimuli = [sen, wrong_sen]

    if not is_encoder:
        return ilm_model.sequence_score(stimuli, reduction=lambda x: x)
    else:
        answers = [mask_1 + ' ' + head + ' ' + mask_2, mask_1 + ' ' + swap_head + ' ' + mask_2]
        return ilm_model.conditional_score(
            prefix=[
                f" {mask_1} ".join([x.strip() for x in sen.split(head)]).strip(),
                f" {mask_1} ".join([x.strip() for x in wrong_sen.split(swap_head)]).strip(),
            ],
            stimuli=answers,
            reduction=lambda x: x
        )
