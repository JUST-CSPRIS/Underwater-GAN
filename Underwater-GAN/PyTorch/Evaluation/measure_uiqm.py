import os
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
from concurrent.futures import ThreadPoolExecutor
## local libs
from uqim_utils import getUIQM


def calculate_uiqm(img_path, im_res=(256, 256)):
    try:
        im = Image.open(img_path)
        # 确保图像尺寸是 window_size 的整数倍
        im = im.resize((im_res[0] - (im_res[0] % 10), im_res[1] - (im_res[1] % 10)))
        im = im.resize(im_res)
        uiqm = getUIQM(np.array(im))
        return uiqm
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def measure_UIQMs_parallel(dir_name, im_res=(256, 256), max_workers=4):
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(calculate_uiqm, img_path, im_res) for img_path in paths]
        for future in futures:
            uiqm = future.result()
            if uiqm is not None:
                uqims.append(uiqm)

    return np.array(uqims)


"""
Get datasets from
 - http://irvlab.cs.umn.edu/resources/euvp-dataset 
 - http://irvlab.cs.umn.edu/resources/ufo-120-dataset 
"""
inp_dir = "/home/xie/xcl/paper/data/SUIM/TEST/images/"
# inp_dir = "/home/xie/xcl/paper/data/UFOtest/TEST/lrd/"
# UIQMs of the distorted input images
inp_uqims = measure_UIQMs_parallel(inp_dir)
print("Input UIQMs >> Mean: {0} std: {1}".format(np.mean(inp_uqims), np.std(inp_uqims)))

## UIQMs of the enhanced output images
gen_dir = "/home/xie/xcl/paper/code/Underwater-GAN/PyTorch/data/funie-suim"
gen_uqims = measure_UIQMs_parallel(gen_dir)
print("Enhanced UIQMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))