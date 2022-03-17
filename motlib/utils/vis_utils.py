import cv2
import numpy as np

# Some simple utility functions 

LINE_SPACING = 50

def write_textlines(
    output, textlines, size=1, offset=(0, 0), fontcolor=(255, 255, 255)
):
    for i, text in enumerate(textlines):
        x = offset[1]
        y = offset[0] + int((i + 1) * size * LINE_SPACING) - 15
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            output, text, (x, y), font, size, fontcolor, 2, cv2.LINE_AA
        )


def draw_text(textlines=[], width=300, fontsize=0.8):
    text_height = int(fontsize * LINE_SPACING * len(textlines))
    text_img = np.zeros((text_height, width, 3), np.uint8)
    write_textlines(text_img, textlines, size=fontsize)
    return text_img


def add_text(img, textlines=[], fontsize=0.8, top=False):
    combined = img
    if len(textlines) > 0:
        text_img = draw_text(textlines, img.shape[1], fontsize)
        if top:
            combined = np.vstack((text_img, img))
        else:
            combined = np.vstack((img, text_img))
    return combined


