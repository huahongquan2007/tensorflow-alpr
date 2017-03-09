import string
import random


def generate_code():
    length = random.choice([8, 9, 10])
    has_dot = random.random() > 0.5
    letters = string.ascii_uppercase
    digits = string.digits
    if has_dot is True:
        code = "{}{}.{}{}.{}{}.{}{}".format(
            random.choice(letters),
            random.choice(letters),
            random.choice(digits),
            random.choice(digits),
            random.choice(letters),
            random.choice(letters+digits),
            random.choice(digits),
            random.choice(digits),
        )
    else:
        code = "{}{} {}{} {}{} {}{}".format(
            random.choice(letters),
            random.choice(letters),
            random.choice(digits),
            random.choice(digits),
            random.choice(letters),
            random.choice(letters+digits),
            random.choice(digits),
            random.choice(digits),
        )
    if length == 8:
        return code
    elif length == 9:
        code += "{}".format(
            random.choice(digits),
        )
        return code
    else:
        code += "{}{}".format(random.choice(digits), random.choice(digits))
        return code

print(generate_code())