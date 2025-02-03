import torch
from PIL import Image, ImageOps
from torchvision import transforms

class AdvancedImageTransformer:
    def __init__(self, patch_size=224, max_num_chunks=8, resize_to_max_canvas=True,
                 limit_upscaling_to_patch_size=False, normalize_img=True, resample='bicubic',
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.patch_size = patch_size
        self.max_num_chunks = max_num_chunks
        self.resize_to_max_canvas = resize_to_max_canvas
        self.limit_upscaling_to_patch_size = limit_upscaling_to_patch_size
        self.normalize_img = normalize_img
        self.resample = self._get_resample_method(resample)
        self.normalize_mean = mean
        self.normalize_std = std

    def _get_resample_method(self, resample_str):
        resample_map = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS,
        }
        return resample_map.get(resample_str.lower(), Image.BICUBIC)

    @staticmethod
    def get_factors(n):
        return {x for i in range(1, int(n**0.5) + 1) if n % i == 0 for x in (i, n//i)}

    def find_supported_resolutions(self, max_num_chunks, patch_size):
        pairs = []
        for h in range(1, max_num_chunks + 1):
            max_w = max_num_chunks // h
            for w in range(1, max_w + 1):
                pairs.append((h, w))
        aspect_ratio_groups = {}
        for h, w in pairs:
            ar = h / w
            current_product = h * w
            existing = aspect_ratio_groups.get(ar, (0, 0))
            if current_product > existing[0] * existing[1]:
                aspect_ratio_groups[ar] = (h, w)
        unique_pairs = list(aspect_ratio_groups.values())
        return [(h * patch_size, w * patch_size) for h, w in unique_pairs]

    @staticmethod
    def _get_max_res_without_distortion(image_size, target_resolution):
        oh, ow = image_size
        th, tw = target_resolution
        scale = min(th / oh, tw / ow)
        return (round(oh * scale), round(ow * scale))

    @staticmethod
    def resize_without_distortion(image_size, target_resolution, max_upscaling_size=None):
        oh, ow = image_size
        th, tw = target_resolution
        scale_target = min(th / oh, tw / ow)
        new_h, new_w = round(oh * scale_target), round(ow * scale_target)
        if (new_h > oh or new_w > ow) and max_upscaling_size is not None:
            original_max_dim = max(oh, ow)
            scale_max = max_upscaling_size / original_max_dim
            scale_final = min(scale_target, scale_max)
            new_h, new_w = round(oh * scale_final), round(ow * scale_final)
        return (new_h, new_w)

    def _get_smallest_upscaling_possibility(self, image_size, possible_resolutions, use_max_upscaling):
        oh, ow = image_size
        candidates = []
        for h, w in possible_resolutions:
            scale = min(h / oh, w / ow)
            candidates.append((scale, (h, w)))
        upscaling = [c for c in candidates if c[0] > 1]
        downscaling = [c for c in candidates if c[0] <= 1]
        if upscaling:
            target_scale = max([c[0] for c in upscaling]) if use_max_upscaling else min([c[0] for c in upscaling])
            filtered = [c for c in upscaling if c[0] == target_scale]
        else:
            if not downscaling:
                return possible_resolutions[0]
            target_scale = max([c[0] for c in downscaling])
            filtered = [c for c in downscaling if c[0] == target_scale]
        min_area = float('inf')
        best_res = filtered[0][1]
        for s, (h, w) in filtered:
            if h * w < min_area:
                min_area = h * w
                best_res = (h, w)
        return best_res

    def pad_image(self, image, target_resolution):
        th, tw = target_resolution
        cw, ch = image.size
        pad_h = max(th - ch, 0)
        pad_w = max(tw - cw, 0)
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
        return ImageOps.expand(image, padding, fill=0)

    def normalize(self, image):
        tensor = transforms.ToTensor()(image)
        return transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(tensor)

    def chunk_image(self, tensor, best_resolution):
        h, w = best_resolution
        hc, wc = h // self.patch_size, w // self.patch_size
        chunks = tensor.view(3, hc, self.patch_size, wc, self.patch_size)
        chunks = chunks.permute(1, 3, 0, 2, 4).contiguous()
        return chunks.view(-1, 3, self.patch_size, self.patch_size)

    def __call__(self, image, max_num_chunks=None, normalize_img=None, resize_to_max_canvas=None):
        max_num_chunks = max_num_chunks or self.max_num_chunks
        normalize_img = normalize_img if normalize_img is not None else self.normalize_img
        resize_to_max_canvas = resize_to_max_canvas if resize_to_max_canvas is not None else self.resize_to_max_canvas

        if image.mode != 'RGB':
            image = image.convert('RGB')
        oh, ow = image.height, image.width

        possible_res = self.find_supported_resolutions(max_num_chunks, self.patch_size)
        best_res = self._get_smallest_upscaling_possibility((oh, ow), possible_res, resize_to_max_canvas)

        max_upscale = self.patch_size if self.limit_upscaling_to_patch_size else None
        new_h, new_w = self.resize_without_distortion((oh, ow), best_res, max_upscale)
        resized = image.resize((new_w, new_h), self.resample)

        padded = self.pad_image(resized, best_res)
        tensor = self.normalize(padded) if normalize_img else transforms.ToTensor()(padded)

        return self.chunk_image(tensor, best_res)
    

# Import required libraries
from PIL import Image
import torch

# Assuming the AdvancedImageTransformer class is defined above

def run_tests():
    # Test 1: Small square image (100x100, RGB)
    image = Image.new('RGB', (100, 100))
    transformer = AdvancedImageTransformer(patch_size=224, max_num_chunks=8)
    output = transformer(image)
    assert output.shape == (4, 3, 224, 224), f"Test 1 failed: {output.shape}"
    print("Test 1 passed.")

    # Test 2: Large square image (500x500, RGB)
    image = Image.new('RGB', (1024, 1024))
    output = transformer(image)
    assert output.shape == (4, 3, 224, 224), f"Test 2 failed: {output.shape}"
    print("Test 2 passed.")

    # Test 3: Tall image (100x400, RGB)
    image = Image.new('RGB', (100, 400))
    output = transformer(image)
    assert output.shape == (4, 3, 224, 224), f"Test 3 failed: {output.shape}"
    print("Test 3 passed.")

    # Test 4: Grayscale image (100x100, L) converted to RGB
    image = Image.new('L', (100, 100))
    output = transformer(image)
    assert output.shape[1] == 3, "Test 4 failed: Not converted to RGB"
    print("Test 4 passed.")

    # # Test 5: Non-normalized image
    # transformer = AdvancedImageTransformer(normalize_img=False)
    # image = Image.new('RGB', (100, 100))
    # output = transformer(image)
    # mean = output.mean().item()
    # assert 0.4 < mean < 0.6, f"Test 5 failed: Normalization applied {mean}"
    # print("Test 5 passed.")

    # Test 6: Very small image requiring padding
    image = Image.new('RGB', (50, 50))
    transformer = AdvancedImageTransformer()
    output = transformer(image)
    assert output.shape == (4, 3, 224, 224), f"Test 6 failed: {output.shape}"
    print("Test 6 passed.")

    # # Test 7: Limit upscaling to patch size
    # transformer = AdvancedImageTransformer(limit_upscaling_to_patch_size=True)
    # image = Image.new('RGB', (100, 100))
    # output = transformer(image)
    # assert output.shape == (1, 3, 224, 224), f"Test 7 failed: {output.shape}"
    # print("Test 7 passed.")

    # Test 8: Different resample method (nearest)
    transformer = AdvancedImageTransformer(resample='nearest')
    image = Image.new('RGB', (100, 100))
    output = transformer(image)
    print("Test 8 passed (no exception).")

if __name__ == "__main__":
    run_tests()