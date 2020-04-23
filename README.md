# reimage
Morph images by maximizing their structural similarity.

## Usage
Just do `reimage.py <from_filename> <to_filename>`. Optional arguments:
* `num_frames`, 128 by default
* `start_blur`, 8.0 by default - how much blurring is done at the start of morphing when comparing images (this is used to speed up convergence, because pixels are attracted to regions where they belong)
* `end_blur`, 0.0 by default - the blurring when the morphing comes to an end
* `choices_per_iteration`, default 16 - how many possible swaps are considered in each iteration (only the best one is chosen)
* `iterations_per_frame`, default 128
* `video_filename`, default `None` - filename to save individual frames as
* `result_filename`, default `None` - filename to save the last frame as
* `no_preview`