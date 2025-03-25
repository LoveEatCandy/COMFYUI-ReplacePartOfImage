import torch
import numpy as np


class ReplacePartOfImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    CATEGORY = "ReplacePartOfImage"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "replace_part_of_image"
    DESCRIPTION = "Replace part of an image with another image"

    def replace_part_of_image(self, image_ref, image_target, mask):
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        mask = mask.cpu()
        batch_size = image_target.size(0)

        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError(
                "ReplacePartOfImage: Use either single reference image or a matching batch of reference images."
            )

        image_ref_np = image_ref.numpy()[0]
        mask_np = mask.numpy()
        mask_np = (mask_np > 0).astype(image_ref_np.dtype)
        mask_np = np.squeeze(mask_np)
        mask_np = np.expand_dims(mask_np, axis=-1)

        out = []
        for i in range(batch_size):
            image_target_np = image_target[i].numpy()

            blended_image = image_target_np * (1 - mask_np) + image_ref_np * mask_np

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
