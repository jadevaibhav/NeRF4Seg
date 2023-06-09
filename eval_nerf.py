import argparse
import os
import time

import imageio
import numpy as np
import torch
import torchvision
import yaml
from tqdm import tqdm

from nerf import (
    CfgNode,
    get_ray_bundle,
    load_blender_data,
    models,
    get_embedding_function,
    run_one_iter_of_nerf,
    render_poses_llff,
    load_llff_data
)

palette = [[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]

def cast_to_image(tensor, dataset_type):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    return img
    # # Map back to shape (3, H, W), as tensorboard needs channels first.
    # return np.moveaxis(img, [-1], [0])

def cast_seg_map(seg,palette,img,opacity=0.5):
    
    seg = seg.detach().cpu()
   
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    #print("seg shape",seg.shape,"color_seg shape", color_seg.shape)#,"palette is",palette)
    for label, color in enumerate(palette):
        #print("label",seg == label)
        color_seg[seg == label, :] = color
    # convert to BGR
    # transpose not required, no need to convert to (3, H, W).
    color_seg = color_seg[..., ::-1]#.transpose((-1,0,1))

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)
    img = np.array(torchvision.transforms.ToPILImage()(img))
    return img

def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = img.clamp(0, 1) * 255
    return img.detach().cpu().numpy().astype(np.uint8)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint / pre-trained model to evaluate.",
    )
    parser.add_argument(
        "--savedir", type=str, help="Save images to this directory, if specified."
    )
    parser.add_argument(
        "--save-disparity-image", action="store_true", help="Save disparity images too."
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    images, poses, render_poses, hwf = None, None, None, None
    i_train, i_val, i_test = None, None, None
    if cfg.dataset.type.lower() == "blender":
        # Load blender dataset
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
        )
        #i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
    elif cfg.dataset.type.lower() == "llff":
        # Load LLFF dataset
        _, poses, bds, render_poses, _, _ = load_llff_data(
            cfg.dataset.basedir, factor=cfg.dataset.downsample_factor,
        )
        hwf = poses[0, :3, -1]

        #Using simple render function to load render_poses
        #render_poses,hwf,poses = render_poses_llff(cfg.dataset.basedir, factor=cfg.dataset.downsample_factor)
        H, W, focal = hwf
        hwf = [int(H), int(W), focal]
        render_poses = torch.from_numpy(render_poses)

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
        model_fine.to(device)

    checkpoint = torch.load(configargs.checkpoint)
    model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    if checkpoint["model_fine_state_dict"]:
        try:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        except:
            print(
                "The checkpoint has a fine-level model, but it could "
                "not be loaded (possibly due to a mismatched config file."
            )
    if "height" in checkpoint.keys():
        hwf[0] = checkpoint["height"]
    if "width" in checkpoint.keys():
        hwf[1] = checkpoint["width"]
    if "focal_length" in checkpoint.keys():
        hwf[2] = checkpoint["focal_length"]

    model_coarse.eval()
    if model_fine:
        model_fine.eval()

    render_poses = render_poses.float().to(device)

    # Create directory to save images to.
    os.makedirs(configargs.savedir, exist_ok=True)
    if configargs.save_disparity_image:
        os.makedirs(os.path.join(configargs.savedir, "disparity"), exist_ok=True)
    
    # creating color palette
    state = np.random.get_state()
    np.random.seed(42)
    # random palette with hard coded num classes 19
    palette19 = np.random.randint(0, 255, size=(19, 3))
    np.random.set_state(state)

    # Evaluation loop
    times_per_image = []
    for i, pose in enumerate(tqdm(render_poses)):
        start = time.time()
        rgb = None, None
        disp = None, None
        with torch.no_grad():
            pose = pose[:3, :4]
            ray_origins, ray_directions = get_ray_bundle(hwf[0], hwf[1], hwf[2], pose)
            rgb_coarse, disp_coarse, _,seg_coarse, rgb_fine, disp_fine, _,seg_fine = run_one_iter_of_nerf(
                hwf[0],
                hwf[1],
                hwf[2],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="validation",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
            rgb = rgb_fine if rgb_fine is not None else rgb_coarse
            seg = seg_fine if seg_fine is not None else seg_coarse
            print("rgb shape",rgb.shape,"seg shape",seg.shape)
            if configargs.save_disparity_image:
                disp = disp_fine if disp_fine is not None else disp_coarse
        times_per_image.append(time.time() - start)
        if configargs.savedir:
            savefile = os.path.join(configargs.savedir, f"{i:04d}.png")
            img = cast_to_image(rgb[..., :3], cfg.dataset.type.lower())
            imageio.imwrite(
                savefile, img
            )

            # saving the segmentation maps rendered on og images
            assert rgb.shape[:-1] == seg.shape[:-1]
            seg = seg.argmax(dim=-1)
            savefile = os.path.join(configargs.savedir, f"{i:04d}_seg.png")
            imageio.imwrite(
                savefile, cast_seg_map(seg[..., :],palette19,img)
            )
            
            if configargs.save_disparity_image:
                savefile = os.path.join(configargs.savedir, "disparity", f"{i:04d}.png")
                imageio.imwrite(savefile, cast_to_disparity_image(disp))
        tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")


if __name__ == "__main__":
    main()
