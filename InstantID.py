import torch
import os
import comfy.utils
import folder_paths
import numpy as np
import math
import cv2
import PIL.Image
from comfy.ldm.modules.attention import optimized_attention
from .resampler import Resampler

from insightface.app import FaceAnalysis

import torchvision.transforms.v2 as T
import torch.nn.functional as F

MODELS_DIR = os.path.join(folder_paths.models_dir, "instantid")
if "instantid" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["instantid"]
folder_paths.folder_names_and_paths["instantid"] = (current_paths, folder_paths.supported_pt_extensions)

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    h, w, _ = image_pil.shape
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

class CrossAttentionPatch:
    # forward for patching
    def __init__(self, weight, instantid, number, cond, uncond, mask=None, sigma_start=0.0, sigma_end=1.0):
        self.weights = [weight]
        self.instantid = [instantid]
        self.conds = [cond]
        self.unconds = [uncond]
        self.number = number
        self.masks = [mask]
        self.sigma_start = [sigma_start]
        self.sigma_end = [sigma_end]

        self.k_key = str(self.number*2+1) + "_to_k_ip"
        self.v_key = str(self.number*2+1) + "_to_v_ip"
    
    def set_new_condition(self, weight, instantid, number, cond, uncond, mask=None, sigma_start=0.0, sigma_end=1.0):
        self.weights.append(weight)
        self.instantid.append(instantid)
        self.conds.append(cond)
        self.unconds.append(uncond)
        self.masks.append(mask)
        self.sigma_start.append(sigma_start)
        self.sigma_end.append(sigma_end)

    def __call__(self, n, context_attn2, value_attn2, extra_options):
        org_dtype = n.dtype
        cond_or_uncond = extra_options["cond_or_uncond"]
        
        sigma = extra_options["sigmas"][0] if 'sigmas' in extra_options else None
        sigma = sigma.item() if sigma else 999999999.9

        q = n
        k = context_attn2
        v = value_attn2
        b = q.shape[0]
        qs = q.shape[1]
        batch_prompt = b // len(cond_or_uncond)
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        _, _, lh, lw = extra_options["original_shape"]
        
        for weight, cond, uncond, instantid, mask, sigma_start, sigma_end in zip(self.weights, self.conds, self.unconds, self.instantid, self.masks, self.sigma_start, self.sigma_end):
            if sigma < sigma_start and sigma > sigma_end:
                k_cond = instantid.ip_layers.to_kvs[self.k_key](cond).repeat(batch_prompt, 1, 1)
                k_uncond = instantid.ip_layers.to_kvs[self.k_key](uncond).repeat(batch_prompt, 1, 1)
                v_cond = instantid.ip_layers.to_kvs[self.v_key](cond).repeat(batch_prompt, 1, 1)
                v_uncond = instantid.ip_layers.to_kvs[self.v_key](uncond).repeat(batch_prompt, 1, 1)

                iid_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
                iid_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

                out_iid = optimized_attention(q, iid_k, iid_v, extra_options["n_heads"])           
                out_iid = out_iid * weight

                if mask is not None:
                    # TODO: needs checking
                    mask_h = lh / math.sqrt(lh * lw / qs)
                    mask_h = int(mask_h) + int((qs % int(mask_h)) != 0)
                    mask_w = qs // mask_h

                    mask_downsample = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bicubic").squeeze(1)

                    # if we don't have enough masks repeat the last one until we reach the right size
                    if mask_downsample.shape[0] < batch_prompt:
                        mask_downsample = torch.cat((mask_downsample, mask_downsample[-1:, :, :].repeat((batch_prompt-mask_downsample.shape[0], 1, 1))), dim=0)
                    # if we have too many remove the exceeding
                    elif mask_downsample.shape[0] > batch_prompt:
                        mask_downsample = mask_downsample[:batch_prompt, :, :]
                    
                    # repeat the masks
                    mask_downsample = mask_downsample.repeat(len(cond_or_uncond), 1, 1)
                    mask_downsample = mask_downsample.view(mask_downsample.shape[0], -1, 1).repeat(1, 1, out.shape[2])

                    out_iid = out_iid * mask_downsample

                out = out + out_iid

        return out.to(dtype=org_dtype)


class InstantID(torch.nn.Module):
    def __init__(self, instantid_model, cross_attention_dim=1280, output_cross_attention_dim=1024, clip_embeddings_dim=512, clip_extra_context_tokens=16):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens

        self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(instantid_model["image_proj"])
        self.ip_layers = To_KV(instantid_model["ip_adapter"])

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4
        )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        #image_prompt_embeds = clip_embed.clone().detach()
        image_prompt_embeds = self.image_proj_model(clip_embed)
        #uncond_image_prompt_embeds = clip_embed_zeroed.clone().detach()
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)

        return image_prompt_embeds, uncond_image_prompt_embeds

class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class To_KV(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = torch.nn.ModuleDict()
        for key, value in state_dict.items():
            k = key.replace(".weight", "").replace(".", "_")
            self.to_kvs[k] = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[k].weight.data = value

def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        patch = CrossAttentionPatch(**patch_kwargs)
        to["patches_replace"]["attn2"][key] = patch
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)

class InstantIDModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "instantid_file": (folder_paths.get_filename_list("instantid"), )}}

    RETURN_TYPES = ("INSTANTID",)
    FUNCTION = "load_model"
    CATEGORY = "InstantID"

    def load_model(self, instantid_file):
        ckpt_path = folder_paths.get_full_path("instantid", instantid_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model
        
        return (model,)

def tensorToNP(image):
    out = torch.clamp(255. * image.detach().cpu(), 0, 255).to(torch.uint8)
    out = out[..., [2, 1, 0]]
    out = out.numpy()
    return out

def extractFeatures(insightface, image, extract_kps=False):
    face_img = tensorToNP(image)
    out = []

    insightface.det_model.input_size = (640,640) # reset the detection size

    for i in range(face_img.shape[0]):
        for size in [(size, size) for size in range(640, 128, -64)]:
            insightface.det_model.input_size = size # TODO: hacky but seems to be working
            face = insightface.get(face_img[i])
            if face:
                face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]

                if extract_kps:
                    out.append(draw_kps(face_img[i], face['kps']))
                else:
                    out.append(torch.from_numpy(face['embedding']).unsqueeze(0))

                if 640 not in size:
                    print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                break

    if out:
        if extract_kps:
            out = torch.stack(T.ToTensor()(out), dim=0).permute([0,2,3,1])
        else:
            out = torch.stack(out, dim=0)
    else:
        out = None

    return out

class InstantIDFaceAnalysis:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"], ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insight_face"
    CATEGORY = "InstantID"

    def load_insight_face(self, provider):
        model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',]) # buffalo_l
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)

class FaceKeypointsPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "faceanalysis": ("FACEANALYSIS", ),
                "image": ("IMAGE", ),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_image"
    CATEGORY = "InstantID"

    def preprocess_image(self, faceanalysis, image):
        face_kps = extractFeatures(faceanalysis, image, extract_kps=True)

        if face_kps is None:
            face_kps = torch.zeros_like(image)
            print(f"\033[33mWARNING: no face detected, unable to extract the keypoints!\033[0m")
            #raise Exception('Face Keypoints Image: No face detected.')

        return (face_kps,)

class ApplyInstantID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instantid": ("INSTANTID", ),
                "insightface": ("FACEANALYSIS", ),
                "image_features": ("IMAGE", ),
                "model": ("MODEL", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,}),
            },
            "optional": {
                "attn_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("MODEL", "POSITIVE", "NEGATIVE", )
    FUNCTION = "apply_instantid"
    CATEGORY = "InstantID"

    def apply_instantid(self, instantid, insightface, image_features, model, positive, negative, weight, start_at, end_at, attn_mask=None):
        self.dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        self.device = comfy.model_management.get_torch_device()
        self.weight = weight

        output_cross_attention_dim = instantid["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        is_sdxl = output_cross_attention_dim == 2048
        cross_attention_dim = 1280
        clip_extra_context_tokens = 16

        face_embed = extractFeatures(insightface, image_features)
        if face_embed is None:
            raise Exception('Feature Extractor: No face detected.')

        clip_embed = face_embed
        clip_embed_zeroed = torch.zeros_like(clip_embed)

        clip_embeddings_dim = face_embed.shape[-1]

        self.instantid = InstantID(
            instantid,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens,
        )

        self.instantid.to(self.device, dtype=self.dtype)

        image_prompt_embeds, uncond_image_prompt_embeds = self.instantid.get_image_embeds(clip_embed.to(self.device, dtype=self.dtype), clip_embed_zeroed.to(self.device, dtype=self.dtype))

        image_prompt_embeds = image_prompt_embeds.to(self.device, dtype=self.dtype)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(self.device, dtype=self.dtype)

        work_model = model.clone()

        sigma_start = work_model.model.model_sampling.percent_to_sigma(start_at)
        sigma_end = work_model.model.model_sampling.percent_to_sigma(end_at)

        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        patch_kwargs = {
            "number": 0,
            "weight": self.weight,
            "instantid": self.instantid,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
            "mask": attn_mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
        }

        if not is_sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("input", id))
                patch_kwargs["number"] += 1
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("output", id))
                patch_kwargs["number"] += 1
            set_model_patch_replace(work_model, patch_kwargs, ("middle", 0))
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                    patch_kwargs["number"] += 1
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                    patch_kwargs["number"] += 1
            for index in range(10):
                set_model_patch_replace(work_model, patch_kwargs, ("middle", 0, index))
                patch_kwargs["number"] += 1

        pos = []
        for t in positive:
            n = [t[0], t[1].copy()]
            n[1]['cross_attn_controlnet'] = image_prompt_embeds.cpu()
            pos.append(n)
        #pos[0][1]['cross_attn_controlnet'] = image_prompt_embeds.cpu()
        
        neg = []
        for t in negative:
            n = [t[0], t[1].copy()]
            n[1]['cross_attn_controlnet'] = uncond_image_prompt_embeds.cpu()
            neg.append(n)
        #neg[0][1]['cross_attn_controlnet'] = uncond_image_prompt_embeds.cpu()

        return(work_model, pos, neg, )

NODE_CLASS_MAPPINGS = {
    "InstantIDModelLoader": InstantIDModelLoader,
    "InstantIDFaceAnalysis": InstantIDFaceAnalysis,
    "ApplyInstantID": ApplyInstantID,
    "FaceKeypointsPreprocessor": FaceKeypointsPreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantIDModelLoader": "Load InstantID Model",
    "InstantIDFaceAnalysis": "InstantID Face Analysis",
    "ApplyInstantID": "Apply InstantID",
    "FaceKeypointsPreprocessor": "Face Keypoints Preprocessor",
}
