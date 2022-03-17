import glob
import numpy as np
from PIL import Image

data_dir = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_real"


if __name__ == "__main__":
    R = []
    G = []
    B = []
    D = []

    # Get statistics on the train set
    images = glob.glob(f"{data_dir}/train/*")
    for RGB_image in images:
        image_name = (RGB_image.split('/')[-1]).split('.')[0]
        depth_image_name = f"{image_name}_d.png"
        depth_image = f"{data_dir}/depth/{depth_image_name}"
        
        depth_value = np.array(
            Image.open(depth_image),
            dtype=np.float32,
        ).flatten()

        D.append(np.mean(depth_value))

        RGB_value = np.array(Image.open(RGB_image).convert("RGB"))
        R.append(np.mean(RGB_value[:, :, 0]))
        G.append(np.mean(RGB_value[:, :, 1]))
        B.append(np.mean(RGB_value[:, :, 2]))
    

    print(f"MEAN VALUE for RGBD: [{np.mean(R)}, {np.mean(G)}, {np.mean(B)}, {np.mean(D)}]")
    print(f"STD VALUE for RGBD: [{np.std(R)}, {np.std(G)}, {np.std(B)}, {np.std(D)}]")

    