import random
import os
import numpy
import math
import cv2
import itertools
import sys
import json
from datetime import datetime
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common


FONT_DIR = "./fonts"
FONT_HEIGHT = 32 # Pixel size to which the chars are resized

OUTPUT_SHAPE = (64, 128)

CHARS = common.CHARS + " -."


def make_char_ims(font_path, output_height):
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]
        if c in "-.":
            im = Image.open("./fonts_special/s{}.png".format(c))
        else:
            im = Image.new("RGBA", (width, height), (0, 0, 0))
            draw = ImageDraw.Draw(im)
            draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


def generate_code():
    length = random.choice([8, 9, 10])
    # length = random.choice([10])
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
            random.choice(common.RANDOM_CHARS),
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
            random.choice(common.RANDOM_CHARS),
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


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def make_affine_transform(from_shape, to_shape, min_scale, max_scale, scale_variation=1.0,
                          rotation_variation=1.0, translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)

    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)
    return out


def generate_plate(font_height, char_ims):
    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2), int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors()

    text_mask = numpy.zeros(out_shape)

    x = h_padding
    y = v_padding
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy: iy + char_im.shape[0], ix: ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +\
             numpy.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")


def generate_real_plate(list_real_data, real_image_data):
    data = random.choice(list_real_data)
    p1, p2, p3, p4 = data['p1'], data['p2'], data['p3'], data['p4']
    top = min(p1['y'], p4['y'])
    bottom = max(p2['y'], p3['y'])
    left = min(p1['x'], p2['x'])
    right = max(p1['x'], p2['x'])

    if right - left > 3 * (bottom - top):
        padding_x = int((right - left) / 4)
        width = padding_x * 2 + (right - left)
        height = int(OUTPUT_SHAPE[0] * width / OUTPUT_SHAPE[1])
        padding_y = int((height - (bottom - top))/2)
    else:
        padding_y = int((bottom - top) / 2)
        height = padding_y * 2 + (bottom - top)
        width = int(OUTPUT_SHAPE[1] * height / OUTPUT_SHAPE[0])
        padding_x = int((width - (right - left))/2)

    if data['path'] in real_image_data:
        image = real_image_data[data['path']]
    else:
        image = cv2.imread(data['path'], cv2.IMREAD_GRAYSCALE)

    shift_width = random.randint(-int(padding_x / 3), int(padding_x / 3))
    shift_height = random.randint(-int(padding_y / 3), int(padding_y / 3))

    img_height, img_width = image.shape[0], image.shape[1]
    new_left = max(0, left - padding_x + shift_width)
    new_right = min(img_width - 1, right + padding_x + shift_width)
    new_top = max(0, top - padding_y + shift_height)
    new_bottom = min(img_height - 1, bottom + padding_y + shift_height)

    plate = numpy.copy(image[new_top: new_bottom, new_left: new_right]) / 255.
    mask = numpy.zeros_like(image)
    cv2.fillPoly(mask, numpy.array([[
                        (p1['x'], p1['y']),
                        (p2['x'], p2['y']),
                        (p3['x'], p3['y']),
                        (p4['x'], p4['y'])]]), 1.0)
    plate_mask = numpy.copy(mask[new_top: new_bottom, new_left: new_right])
    return plate, plate_mask, data['name']


def generate_bg(num_bg_images):
    found = False
    while not found:
        file_name = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            bg = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE) / 255.
            if bg.shape[1] >= OUTPUT_SHAPE[1] and bg.shape[0] >= OUTPUT_SHAPE[0]:
                found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg


def generate_im(char_ims, num_bg_images, list_real_data, real_image_data):
    bg = generate_bg(num_bg_images)

    if random.random() > 0.8:
        plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims)

        M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.6,
                            max_scale=0.875,
                            rotation_variation=1.0,
                            scale_variation=1.5,
                            translation_variation=1.2)

        plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
        plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))
        out = plate * plate_mask + bg * (1.0 - plate_mask)
    else:
        plate, plate_mask, code = generate_real_plate(list_real_data, real_image_data)
        plate = cv2.resize(plate, (bg.shape[1], bg.shape[0]))
        plate_mask = cv2.resize(plate_mask, (bg.shape[1], bg.shape[0]))
        out_of_bounds = False

        if random.random() > 0.8:
            out = plate
        else:
            out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    return out, code, not out_of_bounds


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path, font), FONT_HEIGHT))
    return fonts, font_char_ims


def generate_ims():
    fonts, font_char_imgs = load_fonts(FONT_DIR)
    num_bg_images = len(os.listdir("bgs"))
    list_real_data_path = [os.path.join("data/toll-plaza-a/labels", name) for name in os.listdir("data/toll-plaza-a/labels")]
    list_real_data = list()
    for data_path in list_real_data_path:
        with open(data_path) as fp:
            data_str = fp.read()
            data = json.loads(data_str)
            list_real_data.append(data)
    list_image_path = [os.path.join("data/toll-plaza-a/raw", name) for name in os.listdir("data/toll-plaza-a/raw")]
    real_image_data = dict()
    for image_path in list_image_path:
        real_image_data[image_path] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    while True:
        yield generate_im(font_char_imgs[random.choice(fonts)], num_bg_images, list_real_data, real_image_data)


if __name__ == "__main__":
    os.mkdir("test")
    im_gen = itertools.islice(generate_ims(), int(sys.argv[1]))
    for img_idx, (im, c, p) in enumerate(im_gen):
        file_name = "test/{:08d}_{}_{}.png".format(img_idx, c, "1" if p else "0")
        print(file_name)
        cv2.imwrite(file_name, im * 255.)
