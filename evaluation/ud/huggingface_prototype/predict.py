import torch

from lemma_rule import apply_lemma_rule
from preprocessor import Preprocessor
from utils import load_model


sentences = [
    "One evening I was walking along a path, the city was on one side and the fjord below.",
    "I stopped and looked out over the fjord – the sun was setting, and the clouds turning blood red.",
]
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, model, predict_model = load_model(model_path="HPLT/hplt_gpt_bert_base_3_0_eng_Latn", device=device)
predict_model.eval()
preprocessor = Preprocessor(tokenizer=tokenizer)
batch = preprocessor.preprocess(sentences)
with torch.no_grad():
    lemma_p, upos_p, xpos_p, feats_p, _, __, dep_p, head_p = predict_model(
                                batch["subwords"],
                                batch["alignment"],
                                batch["subword_lengths"],
                                batch["word_lengths"],
                            )
for i in range(len(sentences)):
    for j, form in enumerate(batch["words"][i]):

        lemma_rule = {
            rule_type: model["dataset"].lemma_vocab[rule_type].get(p[i, j, :].argmax().item(),
                       model["dataset"].lemma_vocab[rule_type][-1])
            for rule_type, p in lemma_p.items()
        }
        print(form)
        print(f"lemma {apply_lemma_rule(form, lemma_rule)}")
        print(f"upos {model['dataset'].upos_vocab[upos_p[i, j, :].argmax().item()]}")
        print(f"xpos {model['dataset'].xpos_vocab[xpos_p[i, j, :].argmax().item()]}")
        print(f"feats {model['dataset'].feats_vocab[feats_p[i, j, :].argmax().item()]}")
        print(f"head {head_p[i, j].item()}")
        print(f"deprel {model['dataset'].arc_dep_vocab[dep_p[i, j, :].argmax().item()]}")
        print("_____________________________________________________________________")
