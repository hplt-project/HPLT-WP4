from norsk import mask

ASSETS = (
    (
        'Når det gjelder tida-før og nå, finner jeg de andre forskjellene.',
        'Når det gjelder tida-før og nå-finner jeg andre forskjeller.',
        'Når det gjelder tida-før og nå▁<extra_id_0>finner jeg▁<extra_id_1> andre forskjelle▁<extra_id_2>.</s>',
        'Når det gjelder tida-før og nå▁<extra_id_0>finner jeg▁<extra_id_1> andre forskjelle▁<extra_id_2>.</s>',
        '▁<extra_id_0>, ▁<extra_id_1> de▁<extra_id_2>ne▁<extra_id_3>',
        '▁<extra_id_0>-▁<extra_id_1>▁<extra_id_2>r▁<extra_id_3>',
    ),
    (
        'Når det gjelder tida-før og nå, finner jeg andre forskjeller!',
        'Når det gjelder tida-før og nå-finner jeg andre forskjeller.',
        'Når det gjelder tida-før og nå▁<extra_id_0>finner jeg andre forskjeller▁<extra_id_1></s>',
        'Når det gjelder tida-før og nå▁<extra_id_0>finner jeg andre forskjeller▁<extra_id_1></s>',
        '▁<extra_id_0>, ▁<extra_id_1>!▁<extra_id_2>',
        '▁<extra_id_0>-▁<extra_id_1>.▁<extra_id_2>',
    ),
(
        'Når det gjelder tida-før og nå, finner jeg andre forskjeller',
        'Når det gjelder tida-før og nå-finner jeg andre forskjeller.',
        'Når det gjelder tida-før og nå▁<extra_id_0>finner jeg andre forskjeller▁<extra_id_1></s>',
        'Når det gjelder tida-før og nå▁<extra_id_0>finner jeg andre forskjeller▁<extra_id_1></s>', # the idea is that if it is wrong, it should lower the whole probability much enough
        '▁<extra_id_0>, ▁<extra_id_1>▁<extra_id_2>',  # even if the sentence is longer
        '▁<extra_id_0>-▁<extra_id_1>.▁<extra_id_2>',
    ),
    (

        'Hvis det gjelder tida-før og nå, finner jeg de andre forskjellene.',
        'Når det gjelder tida-før og nå-finner jeg andre forskjeller.',
        '▁<extra_id_0> det gjelder tida-før og nå▁<extra_id_1>finner jeg▁<extra_id_2> andre forskjelle▁<extra_id_3>.</s>',
        '▁<extra_id_0> det gjelder tida-før og nå▁<extra_id_1>finner jeg▁<extra_id_2> andre forskjelle▁<extra_id_3>.</s>',
        'Hvis▁<extra_id_0>, ▁<extra_id_1> de▁<extra_id_2>ne▁<extra_id_3>',
        'Når▁<extra_id_0>-▁<extra_id_1>▁<extra_id_2>r▁<extra_id_3>',
    ),

)


def test_norsk():
    for sample in ASSETS:
        in_1, in_2, out_1, out_2 = mask(sample[0], sample[1])
        assert in_1 == sample[2]
        assert in_2 == sample[3]
        assert out_1 == sample[4]
        assert out_2 == sample[5]
