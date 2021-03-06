import torch
import numpy as np

from data import transforms


def get_mask(resolution, return_mask=False, R=4, p_m=False):
    if R == 8 and resolution == 384:
        a = np.array(
            [1, 24, 45, 64, 81, 97, 111, 123, 134, 144, 153, 161, 168, 175, 181, 183, 184, 185, 186, 187, 188, 189, 190,
             191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 205, 211, 218, 225, 233, 242, 252, 263, 275, 289,
             305, 322, 341, 362])
        m = np.zeros((384, 384))
        m[:, a] = True
        m[:, 183:201] = True


    if R == 4 and resolution == 384:
        a = np.array(
            [0, 10, 19, 28, 37, 46, 54, 61, 69, 76, 83, 89, 95, 101, 107, 112, 118, 122, 127, 132, 136, 140, 144, 148,
             151, 155, 158, 161, 164,
             167, 170, 173, 176, 178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211, 214, 217, 220,
             223, 226, 229, 233, 236,
             240, 244, 248, 252, 257, 262, 266, 272, 277, 283, 289, 295, 301, 308, 315, 323, 330, 338, 347, 356, 365,
             374])
        m = np.zeros((384, 384))
        m[:, a] = True
        m[:, 176:208] = True
    elif R == 4:
        a = np.array(
            [1, 10, 18, 25, 31, 37, 42, 46, 50, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
             76, 80, 84, 88, 93, 99, 105, 112, 120])

        m = np.zeros((128, 128))
        m[:, a] = True
        m[:, 56:73] = True

    if p_m:
        pass

    samp = m
    numcoil = 16
    mask = transforms.to_tensor(np.tile(samp, (numcoil, 1, 1)).astype(np.float32))
    mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    return mask if return_mask else np.where(m == 1)
