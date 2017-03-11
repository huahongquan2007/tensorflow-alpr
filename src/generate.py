import random
import os
import numpy
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common

FONT_DIR = "./fonts"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized

OUTPUT_SHAPE = (64, 128)

CHARS = common.CHARS + " "


def make_char_ims(font_path, output_height):
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


def generate_code():
    length = random.choice([8, 9, 10])
    has_dot = random.random() > 0.5
    if has_dot is True:
        is_dash = random.random() > 0.5
        if is_dash:
            format_string = "{}{}-{}{}-{}{}-{}{}"
        else:
            format_string = "{}{}.{}{}.{}{}.{}{}"
        code = format_string.format(
            random.choice(common.LETTERS),
            random.choice(common.LETTERS),
            random.choice(common.DIGITS),
            random.choice(common.DIGITS),
            random.choice(common.LETTERS),
            random.choice(common.CHARS),
            random.choice(common.DIGITS),
            random.choice(common.DIGITS),
        )
    else:
        code = "{}{} {}{} {}{} {}{}".format(
            random.choice(common.LETTERS),
            random.choice(common.LETTERS),
            random.choice(common.DIGITS),
            random.choice(common.DIGITS),
            random.choice(common.LETTERS),
            random.choice(common.CHARS),
            random.choice(common.DIGITS),
            random.choice(common.DIGITS),
        )
    if length == 8:
        return code
    elif length == 9:
        code += "{}".format(
            random.choice(common.DIGITS),
        )
        return code
    else:
        code += "{}{}".format(random.choice(common.DIGITS), random.choice(common.DIGITS))
        return code


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path, font), FONT_HEIGHT))
    return fonts, font_char_ims

print(generate_code())
ims = make_char_ims("./fonts/UKNumberPlate.ttf", FONT_HEIGHT)
print(ims)