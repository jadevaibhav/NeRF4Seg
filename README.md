We propose a novel method of using NeRF models to generalize segmentation masks learned on the data used to train the NeRF model onto a scene. Our approach falls under the paradigm of parallelly learning a class label for each pixel value when the model learns a 3D reconstruction from multiview RGB images. Please refer to [the report](/blob/main/blob/main/NeRFing%20the%20boundaries.pdf) for more details of our methodology.

## How to train your NeRF super-quickly!

To train a "full" NeRF model (i.e., using 3D coordinates as well as ray directions, and the hierarchical sampling procedure), first setup dependencies. 

### Option 1: Using pip

In a new `conda` or `virtualenv` environment, run

```bash
pip install -r requirements.txt
```

### Option 2: Using conda

Use the provided `environment.yml` file to install the dependencies into an environment named `nerf` (edit the `environment.yml` if you wish to change the name of the `conda` environment).

```bash
conda env create
conda activate nerf
```

### Adding segmentation masks for training

Generate and store the segmentation mask for all train images in single .npy file(as one-hot encoding) in root folder. We have provided our segmentation mask 8x downscaled resolution of room images for reproducibility.

### Run training!

Once everything is setup, to run experiments, first edit `config/room.yml` to specify your own parameters.

The training script can be invoked by running
```bash
python train_nerf.py --config config/room.yml
```

### Optional: Resume training from a checkpoint

Optionally, if resuming training from a previous checkpoint, run
```bash
python train_nerf.py --config config/lego.yml --load-checkpoint path/to/checkpoint.ckpt
```

### Optional: Cache rays from the dataset

An optional, yet simple preprocessing step of caching rays from the dataset results in substantial compute time savings (reduced carbon footprint, yay!), especially when running multiple experiments. It's super-simple: run
```bash
python cache_dataset.py --datapath cache/nerf_synthetic/lego/ --halfres False --savedir cache/legocache/legofull --num-random-rays 8192 --num-variations 50
```

This samples `8192` rays per image from the `lego` dataset. Each image is `800 x 800` (since `halfres` is set to `False`), and `500` such random samples (`8192` rays each) are drawn per image. The script takes about 10 minutes to run, but the good thing is, this needs to be run only once per dataset.

> **NOTE**: Do NOT forget to update the `cachedir` option (under `dataset`) in your config (.yml) file!
