def min_edit_script(source, target, allow_copy):
    a = [[(len(source) + len(target) + 1, None)] * (len(target) + 1) for _ in range(len(source) + 1)]
    for i in range(0, len(source) + 1):
        for j in range(0, len(target) + 1):
            if i == 0 and j == 0:
                a[i][j] = (0, "")
            else:
                if allow_copy and i and j and source[i - 1] == target[j - 1] and a[i-1][j-1][0] < a[i][j][0]:
                    a[i][j] = (a[i-1][j-1][0], a[i-1][j-1][1] + "→")
                if i and a[i-1][j][0] < a[i][j][0]:
                    a[i][j] = (a[i-1][j][0] + 1, a[i-1][j][1] + "-")
                if j and a[i][j-1][0] < a[i][j][0]:
                    a[i][j] = (a[i][j-1][0] + 1, a[i][j-1][1] + "+" + target[j - 1])
    return a[-1][-1][1]


def gen_lemma_rule(form, lemma, allow_copy):
    best, best_form, best_lemma = 0, 0, 0
    for l in range(len(lemma)):
        for f in range(len(form)):
            cpl = 0
            while f + cpl < len(form) and l + cpl < len(lemma) and form[f + cpl].lower() == lemma[l + cpl].lower():
                cpl += 1
            if cpl > best:
                best = cpl
                best_form = f
                best_lemma = l

    if not best:
        return {"case": None, "prefix": None, "suffix": None, "absolute": "a" + lemma}

    prefix_rule = min_edit_script(form[:best_form].lower(), lemma[:best_lemma].lower(), allow_copy)
    suffix_rule = min_edit_script(form[best_form + best:].lower(), lemma[best_lemma + best:].lower(), allow_copy)

    if lemma.islower():
        return {"case": "lower", "prefix": prefix_rule, "suffix": suffix_rule, "absolute": "relative"}

    generated_lemma = apply_lemma_rule(form, {"case": "lower", "prefix": prefix_rule, "suffix": suffix_rule, "absolute": "relative"}, apply_casing=False)
    if generated_lemma == lemma:
        return {"case": "keep", "prefix": prefix_rule, "suffix": suffix_rule, "absolute": "relative"}

    previous_case = -1
    lemma_casing = ""
    for i, c in enumerate(lemma):
        case = "↑" if c.lower() != c else "↓"
        if case != previous_case:
            lemma_casing += "{}{}{}".format("¦" if lemma_casing else "", case, i if i <= len(lemma) // 2 else i - len(lemma))
        previous_case = case     
    
    return {"case": lemma_casing, "prefix": prefix_rule, "suffix": suffix_rule, "absolute": "relative"}


def apply_lemma_rule(form, lemma_rule, apply_casing=True):
    if lemma_rule["absolute"].startswith("a"):
        return lemma_rule["absolute"][1:]

    if any(rule is None for rule in lemma_rule.values()):
        return form

    rules, rule_sources = (lemma_rule["prefix"], lemma_rule["suffix"]), []
    for rule in rules:
        source, i = 0, 0
        while i < len(rule):
            if rule[i] == "→" or rule[i] == "-":
                source += 1
            else:
                assert rule[i] == "+"
                i += 1
            i += 1
        rule_sources.append(source)

    try:
        lemma, form_offset = "", 0
        for i in range(2):
            j, offset = 0, (0 if i == 0 else len(form) - rule_sources[1])
            while j < len(rules[i]):
                if rules[i][j] == "→":
                    lemma += form[offset]
                    offset += 1
                elif rules[i][j] == "-":
                    offset += 1
                else:
                    assert(rules[i][j] == "+")
                    lemma += rules[i][j + 1]
                    j += 1
                j += 1
            if i == 0:
                lemma += form[rule_sources[0] : len(form) - rule_sources[1]]
    except:
        lemma = form

    if not apply_casing:
        return lemma
    
    if lemma_rule["case"] == "lower":
        return lemma.lower()
    elif lemma_rule["case"] == "keep":
        return lemma

    lemma = lemma.lower()
    for rule in lemma_rule["case"].split("¦"):
        if rule == "↓0": continue # The lemma is lowercased initially
        if not rule: continue # Empty lemma might generate empty casing rule
        case, offset = rule[0], int(rule[1:])
        lemma = lemma[:offset] + (lemma[offset:].upper() if case == "↑" else lemma[offset:].lower())

    return lemma
