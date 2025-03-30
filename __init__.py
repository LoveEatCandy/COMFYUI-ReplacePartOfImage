import torch


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
        image_ref = image_ref.to(torch.float32)
        image_target = image_target.to(torch.float32)

        batch_size = image_target.size(0)

        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError(
                "ReplacePartOfImage: Please use single reference image or a batch of the same size as the target image."
            )

        if mask is not None and mask.size(0) > 1 and mask.size(0) != batch_size:
            raise ValueError(
                "ReplacePartOfImage: Please use single mask or a batch of the same size as the target image."
            )

        first_mask = None
        if mask is not None and mask.size(0) == 1:
            first_mask = mask.squeeze(0).unsqueeze(-1).to(torch.float32)
            first_mask = (first_mask > 0.5).to(image_ref.dtype)

        out = []
        for i in range(batch_size):
            image_ref_i = image_ref[i] if image_ref.size(0) > 1 else image_ref[0]
            image_target_i = image_target[i]
            if mask is not None and mask.size(0) > 1:
                mask_i = mask[i]
                mask_i = mask_i.squeeze(0).unsqueeze(-1).to(torch.float32)
                mask_i = (mask_i > 0.5).to(image_ref.dtype)
            else:
                mask_i = first_mask

            ref_shape = image_ref_i.shape
            target_shape = image_target_i.shape

            if left_top_x > target_shape[1]:
                raise ValueError(
                    f"left_top_x({left_top_x}) must be smaller than target image width({target_shape[1]})"
                )
            if left_top_y > target_shape[0]:
                raise ValueError(
                    f"left_top_y({left_top_y}) must be smaller than target image height({target_shape[0]})"
                )

            width = min(ref_shape[1], target_shape[1] - left_top_x)
            height = min(ref_shape[0], target_shape[0] - left_top_y)

            current_ref = image_ref_i[:height, :width, :]
            part_of_target = image_target_i[
                left_top_y : left_top_y + height, left_top_x : left_top_x + width, :
            ]

            if mask_i is not None:
                current_mask = mask_i[:height, :width, :]
                part_of_blended_image = (
                    part_of_target * (1 - current_mask) + current_ref * current_mask
                )
            else:
                part_of_blended_image = current_ref

            blended_image = image_target_i.clone()
            blended_image[
                left_top_y : left_top_y + height, left_top_x : left_top_x + width, :
            ] = part_of_blended_image

            out.append(blended_image)

        out = torch.stack(out, dim=0).clamp(0, 1)  # 保持数据范围
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
