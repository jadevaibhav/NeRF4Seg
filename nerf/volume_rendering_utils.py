import torch

from .nerf_helpers import cumprod_exclusive


def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
):
    # TESTED
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(radiance_field[..., :3])
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    depth_map = weights * depth_values
    depth_map = depth_map.sum(dim=-1)
    # depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def volume_render_radiance_field_with_seg(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    is_color=False):
    """
    v1: without using color rendering for segmentation(baseline)
    v2: with color rendering
    """
    #distance from the last(distance of the far end) is infinity
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    #normalized ray directions
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(radiance_field[..., :3])
    noise = 0.0
    # added noise for regularization in predicted density
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3:].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
        
    # seperate sigma and alpha value for each segmentation class
    seg_map,weights = seg_3d(radiance_field,dists,noise,is_color=is_color)
    if weights is None:
        return None,None,None,None,None,seg_map
    
    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    depth_map = weights * depth_values
    depth_map = depth_map.sum(dim=-1)
    # depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map, seg_map

def seg_3d(radiance_field,dists,noise,is_color=True):
    """Transforms model's predictions to semantic labels.
        Args:
          radiance_field: [num_rays, num_samples along ray, 3+num_classes]. Prediction from model.
          dists: [num_rays, num_samples along ray]. Integration time.
          noise: [num_rays, 3]. Direction of each ray.
        Returns:
          seg_map: semantic segmentation map
          weights: weights to use for color rendering
        """
    #segmentation mask rendering
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3:] + noise)
   
    alpha = 1.0 - torch.exp(-sigma_a * dists.unsqueeze(-1))
    # dim -2 is num_samples along the ray, along which cumprod is taken
    seg_map = alpha * cumprod_exclusive(1.0 - alpha + 1e-10,dim=-2)
    seg_map = torch.nn.functional.softmax(seg_map,dim=-1)

    #For color rendering, we sum the sigma for all classes
    weights = None
    if is_color:
        print("RF and noise",torch.sum(radiance_field[..., 3:],dim=-1,keepdim=True).shape,noise.shape)
        sigma_a = torch.nn.functional.relu(torch.sum(radiance_field[..., 3:],dim=-1) + noise.squeeze(-1))
        alpha = 1.0 - torch.exp(-sigma_a * dists)
        weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    return seg_map, weights
    