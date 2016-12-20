# https://www.reddit.com/r/dailyprogrammer/comments/10pf0j/9302012_challenge_102_easy_dice_roller/

def roller(spec: str) -> int:
    n, spec_tail = spec.split('d')
    bonus_plus_case = spec_tail.split('+')
    bonus_minus_case = spec_tail.split('-')
    if len(bonus_plus_case) == 2:
        pass  # TODO
    elif len(bonus_minus_case) == 2:
        pass # TODO
    else:
        pass
