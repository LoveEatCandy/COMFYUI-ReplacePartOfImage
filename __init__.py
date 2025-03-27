import torch
import numpy as np


class ReplacePartOfImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK", {"default": None}),
                "left_top_x": ("INT", {"default": 0, "min": 0, "step": 1}),
                "left_top_y": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    CATEGORY = "ReplacePartOfImage"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "replace_part_of_image"
    DESCRIPTION = "Replace part of an image with another image"

    def replace_part_of_image(
        self, image_ref, image_target, mask=None, left_top_x=0, left_top_y=0
    ):
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        image_ref = image_ref.squeeze()
        image_target = image_target.squeeze()
        batch_size = image_target.size(0)

        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError(
                "ReplacePartOfImage: Please use single reference image or a batch of the same size as the target image."
            )

        image_ref_np = image_ref[0].numpy()

        mask_np = None
        if mask is not None:
            if mask.dtype == torch.float16:
                mask = mask.to(torch.float32)
            mask = mask.cpu()
            mask_np = mask.numpy()
            mask_np = (mask_np > 0).astype(image_ref_np.dtype)
            mask_np = np.squeeze(mask_np)
            mask_np = np.expand_dims(mask_np, axis=-1)

        out = []
        for i in range(batch_size):
            image_target_np = image_target[i].numpy()
            image_ref_np = (
                image_ref[i].numpy() if image_ref.size(0) > 1 else image_ref_np
            )

            ref_shape = image_ref_np.shape
            if mask_np is not None and mask_np.shape[:2] != ref_shape[:2]:
                raise ValueError(
                    f"ReplacePartOfImage: Mask({mask_np.shape[:2]}) and reference({ref_shape[:2]}) image must have the same shape."
                )

            target_shape = image_target_np.shape
            if left_top_x > target_shape[1]:
                raise ValueError(
                    f"left_top_x({left_top_x}) must smaller than target image width({target_shape[1]})"
                )
            if left_top_y > target_shape[0]:
                raise ValueError(
                    f"left_top_y({left_top_y}) must smaller than target image height({target_shape[0]})"
                )

            width, height = (
                min(ref_shape[1], target_shape[1] - left_top_x),
                min(ref_shape[0], target_shape[0] - left_top_y),
            )

            current_ref_np = image_ref_np[:height, :width, :]
            current_mask_np = None
            if mask_np is not None:
                current_mask_np = mask_np[:height, :width, :]

            if current_mask_np is None:
                part_of_blended_image = current_ref_np
            else:
                part_of_target = image_target_np[
                    left_top_y : left_top_y + height, left_top_x : left_top_x + width, :
                ]
                part_of_blended_image = (
                    part_of_target * (1 - current_mask_np)
                    + current_ref_np * current_mask_np
                )

            blended_image = image_target_np.copy()
            blended_image[
                left_top_y : left_top_y + height, left_top_x : left_top_x + width, :
            ] = part_of_blended_image

            out.append(torch.tensor(blended_image))

        out = torch.stack(out, dim=0).to(torch.float32)
        return (out,)


NODE_CONFIG = {
    "ReplacePartOfImage": {
        "class": ReplacePartOfImage,
        "name": "Replace Part Of Image",
    },
}


def generate_node_mappings(node_config):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get(
            "name", node_info["class"].__name__
        )

    return node_class_mappings, node_display_name_mappings


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
