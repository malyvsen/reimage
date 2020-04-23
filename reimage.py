import numpy as np
from argparse import ArgumentParser
import skimage
from tqdm import trange
import cv2


parser = ArgumentParser()
parser.add_argument('from_filename', type=str)
parser.add_argument('to_filename', type=str)
parser.add_argument('--num_frames', type=int, default=128)
parser.add_argument('--start_blur', type=float, default=8.0)
parser.add_argument('--end_blur', type=float, default=0.0)
parser.add_argument('--choices_per_iteration', type=int, default=16)
parser.add_argument('--iterations_per_frame', type=int, default=8192)
parser.add_argument('--video_filename', type=str, default=None)
parser.add_argument('--result_filename', type=str, default=None)
parser.add_argument('--no_preview', action='store_true')
args = parser.parse_args()


from_image = skimage.io.imread(args.from_filename)
to_image = skimage.io.imread(args.to_filename)

if from_image.shape != to_image.shape:
    print('Image shapes must match exactly!')
    print(f'Provided shapes: {from_image.shape}, {to_image.shape}')
    exit()


def random_swap(image, blur):
    position = np.array((np.random.randint(1, image.shape[0] - 1), np.random.randint(1, image.shape[1] - 1)))
    direction = np.random.randint(4)
    vector = np.array((1, 0) if direction == 0 else (0, 1) if direction == 1 else (-1, 0) if direction == 2 else (0, -1))

    result = np.copy(image)
    result[position[0], position[1]] = image[position[0]+vector[0], position[1]+vector[1]]
    result[position[0]+vector[0], position[1]+vector[1]] = image[position[0], position[1]]

    box_half_size = max(int(blur * 2), 4)
    position_clipped = np.clip(position, (box_half_size, box_half_size), (image.shape[0] - box_half_size, image.shape[1] - box_half_size))
    box_min = position_clipped - np.array((box_half_size, box_half_size))
    box_max = position_clipped + np.array((box_half_size, box_half_size))
    image_cut = image[box_min[0]:box_max[0], box_min[1]:box_max[1]]
    result_cut = result[box_min[0]:box_max[0], box_min[1]:box_max[1]]
    to_cut = to_image[box_min[0]:box_max[0], box_min[1]:box_max[1]]

    image_filtered = skimage.filters.gaussian(image_cut, sigma=blur, truncate=2.0, multichannel=True)
    result_filtered = skimage.filters.gaussian(result_cut, sigma=blur, truncate=2.0, multichannel=True)
    to_filtered = skimage.filters.gaussian(to_cut, sigma=blur, truncate=2.0, multichannel=True)
    old_ssim = skimage.measure.compare_ssim(image_filtered, to_filtered, multichannel=True)
    new_ssim = skimage.measure.compare_ssim(result_filtered, to_filtered, multichannel=True)

    return result, new_ssim - old_ssim


reimage = np.copy(from_image)
for frame in trange(args.num_frames):
    if not args.no_preview:
        cv2.imshow('reimage', cv2.cvtColor(reimage, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    if args.video_filename is not None:
        skimage.io.imsave(f'{args.video_filename}_{frame}.png', reimage)
    blur = np.interp(frame, [0, args.num_frames - 1], [args.start_blur, args.end_blur])
    for iteration in trange(args.iterations_per_frame):
        iterator = (random_swap(reimage, blur=blur) for i in range(args.choices_per_iteration))
        reimage, ssim_gain = max(iterator, key=lambda x: x[1])


cv2.destroyAllWindows()
if args.result_filename is not None:
    skimage.io.imsave(args.result_filename, reimage)
