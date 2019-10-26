import gc
import random
import numpy as np

from tqdm import tqdm
from PIL import Image

from skimage.transform import resize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#----------------------------------------------------------------------
# Scale and visualize the latent vectors
def plot_dataset3d(X, y, save=None):
    if len(y.shape)> 1:
        y = y[:,0]
    plt.cla()
    print('data size {}'.format(X.shape))

    fig = plt.figure(figsize=(27, 18), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    uni_y = len(np.unique(y))
    for yi in tqdm(range(uni_y)):
        ax.scatter(X[:, 0][y == yi], X[:, 1][y == yi], X[:, 2][y == yi], color=plt.cm.Set1(yi / uni_y), marker='o')

    #ax.axis('off')
    if save is not None:
        print('Saving Image {} ...'.format(save))
        plt.title('epoch ' + save.split('.')[0].split()[-1], fontdict={'fontsize': 20}, loc='left')
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
    del X, y, fig, ax
    gc.collect()

def plot_dataset(X, y, save=None):
    if len(y.shape)> 1:
        y = y[:,0]
    plt.cla()
    print('data size {}'.format(X.shape))

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(27, 18), dpi=100)
    ax = plt.subplot(111)

    uni_y = len(np.unique(y))
    for yi in tqdm(range(uni_y)):
        ax.scatter(X[:, 0][y == yi], X[:, 1][y == yi], color=plt.cm.Set1(yi / uni_y), marker='o')

    plt.xticks([]), plt.yticks([])
    #for item in [fig, ax]:
    #    item.patch.set_visible(False)
    ax.axis('off')

    if save is not None:
        print('Saving Image {} ...'.format(save))
        plt.title('epoch '+ save.split('.')[0].split()[-1], fontdict={'fontsize': 20}, loc='left')
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
    del X, y, fig, ax
    gc.collect()

def plot_samples(samples, scale=10, save=None):

    im = merge(samples, (10,10))

    fig_width = int(im.shape[0] * scale)
    fig_height = int(im.shape[1] * scale)

    im = resize(im, (fig_width, fig_height), anti_aliasing=True)

    fig = plt.figure(dpi=150)
    ax = plt.subplot(111)
    plt.imshow(im)
    for item in [fig, ax]:
        item.patch.set_visible(False)
    plt.axis('off')

    if save is not None:
        print('Saving Image ', save)
        plt.title('epoch '+ save.split('.')[0].split()[-1], fontdict={'fontsize': 8}, loc='left')
        plt.savefig(save)
        plt.close()
    del im, samples, fig, ax
    gc.collect()

def pick_n(X, n):
    samples = list()
    for _ in range(n):
         samples.append(random.randint(0,len(X)-1))
    return X[:n]

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    images = pick_n(images, 100)
    if (images.shape[3] in (3,4)):
        c = images.shape[-1:][0]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def resize_gif(path, save_as=None, resize_to=None):
    """
    Resizes the GIF to a given length:

    Args:
        path: the path to the GIF file
        save_as (optional): Path of the resized gif. If not set, the original gif will be overwritten.
        resize_to (optional): new size of the gif. Format: (int, int). If not set, the original GIF will be resized to
                              half of its size.
    """
    all_frames = extract_and_resize_frames(path, resize_to)

    if not save_as:
        save_as = path

    if len(all_frames) == 1:
        print("Warning: only 1 frame found")
        all_frames[0].save(save_as, optimize=True)
    else:
        all_frames[0].save(save_as, optimize=True, save_all=True, append_images=all_frames[1:], loop=1000)


def analyseImage(path):
    """
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode
    before processing all frames.
    """
    im = Image.open(path)
    results = {
        'size': im.size,
        'mode': 'full',
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results


def extract_and_resize_frames(path, resize_to=None):
    """
    Iterate the GIF, extracting each frame and resizing them

    Returns:
        An array of all frames
    """
    mode = analyseImage(path)['mode']

    im = Image.open(path)

    if not resize_to:
        resize_to = (im.size[0] // 2, im.size[1] // 2)

    i = 0
    p = im.getpalette()
    last_frame = im.convert('RGBA')

    all_frames = []

    try:
        while True:
            # print("saving %s (%s) frame %diag, %s %s" % (path, mode, i, im.size, im.tile))

            '''
            If the GIF uses local colour tables, each frame will have its own palette.
            If not, we need to apply the global palette to the new frame.
            '''
            if not im.getpalette():
                im.putpalette(p)

            new_frame = Image.new('RGBA', im.size)

            '''
            Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
            If so, we need to construct the new frame by pasting it on top of the preceding frames.
            '''
            if mode == 'partial':
                new_frame.paste(last_frame)

            new_frame.paste(im, (0, 0), im.convert('RGBA'))

            new_frame.thumbnail(resize_to, Image.ANTIALIAS)
            all_frames.append(new_frame)

            i += 1
            last_frame = new_frame
            im.seek(im.tell() + 1)
    except EOFError:
        pass

    return all_frames




