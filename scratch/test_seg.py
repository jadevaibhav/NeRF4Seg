from nerf import volume_render_radiance_field,predict_and_render_radiance
from nerf import run_network,get_minibatches,get_ray_bundle,positional_encoding,ndc_rays
from nerf import sample_pdf_2 as sample_pdf
import torch

tform_cam2world = torch.eye(4)
H,W = 5,5
focal = 10
rays_o, rays_d = get_ray_bundle(H,W,focal_length=focal,tform_cam2world=tform_cam2world)
print("get_ray_bundle:rays_o, rays_d")
print("rays_o:",rays_o.shape)
print("rays_d:",rays_d.shape)

r_o,r_d = ndc_rays(H,W,focal,1,rays_o=rays_o,rays_d=rays_d)

# sample arguements
#Caching dataset: python3 cache_dataset.py --datapath data/lego/ --type blender --blender-half-res True --savedir cache/legocache/legohalf --num-random-rays 8192 --num-variations 50
#Eval script: 

#python3 eval_nerf.py --config pretrained/lego-lowres/config.yml --checkpoint pretrained/lego-lowres/checkpoint199999.ckpt --savedir cache/rendered/lego-lowres         