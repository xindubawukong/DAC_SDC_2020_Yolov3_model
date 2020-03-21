import cv2
import matplotlib.pyplot as plt
import os


# Read image to RGB format.
def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(img, path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def show_image(img, figsize=(10, 10), gray=False):
    plt.figure(figsize=figsize)
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def GetLogger(name='mylogger', output_file=None):
    import logging
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # handler = logging.StreamHandler()
    # handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('[%(asctime)s %(filename)s %(levelname)s] %(message)s')
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)

    if output_file is not None:
        if os.path.exists(output_file):
            os.system('rm -rf ' + output_file)
        handler = logging.FileHandler(filename=output_file, mode='a', encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s %(filename)s %(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
