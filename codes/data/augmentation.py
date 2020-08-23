import numpy as np
import cv2

def t_random(min=0, max=1):
    return min + (max - min) * np.random.rand()


def t_randint(min, max):
    return np.random.randint(low=min, high=max)

class augCompose(object):

    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, img, mask):

        if self.transforms is not None:
            for op, prob in self.transforms:
                if t_random() <= prob:
                    img, mask = op(img, mask)

        return img, mask

def RandomFlip(img, mask, FLIP_LEFT_RIGHT=True, FLIP_TOP_BOTTOM=True):

    if FLIP_LEFT_RIGHT and t_random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    if FLIP_TOP_BOTTOM and t_random() < 0.5:
        img = cv2.flip(img, 0)
        mask = cv2.flip(mask, 0)
    return img, mask


def RandomBlur(img, mask):

    r = 5

    if t_random() < 0.2:
        return cv2.GaussianBlur(img,(r,r),0), mask

    if t_random() < 0.15:
        return cv2.blur(img,(r,r)), mask

    if t_random() < 0.1:
        return cv2.medianBlur(img,r), mask


    return img, mask

def RandomColorJitter(img, mask, brightness=32, contrast=0.5, saturation=0.5, hue=0.1,
                        prob=0.5):
    if brightness != 0 and t_random() > prob:
        img = _Brightness(img, delta=brightness)
    if contrast != 0 and t_random() > prob:
        img = _Contrast(img, var=contrast)
    if saturation != 0 and t_random() > prob:
        img = _Saturation(img, var=saturation)
    if hue != 0 and t_random() > prob:
        img = _Hue(img, var=hue)

    return img, mask


def _Brightness(img, delta=32):
    img = img.astype(np.float32) + t_random(-delta, delta)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _Contrast(img, var=0.3):
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    alpha = 1.0 + t_random(-var, var)
    img = alpha * img.astype(np.float32) + (1 - alpha) * gs
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _Hue(img, var=0.05):
    var = t_random(-var, var)
    to_HSV, from_HSV = [
        (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
        (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR)][t_randint(0, 2)]
    hsv = cv2.cvtColor(img, to_HSV).astype(np.float32)

    hue = hsv[:, :, 0] / 179. + var
    hue = hue - np.floor(hue)
    hsv[:, :, 0] = hue * 179.

    img = cv2.cvtColor(hsv.astype('uint8'), from_HSV)
    return img


def _Saturation(img, var=0.3):
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gs = np.expand_dims(gs, axis=2)
    alpha = 1.0 + t_random(-var, var)
    img = alpha * img.astype(np.float32) + (1 - alpha) * gs.astype(np.float32)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


