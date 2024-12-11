# from transformers import AutoTokenizer, BitsAndBytesConfig
# from llava.model import LlavaLlamaForCausalLM
# import torch

# def initialize_model(model_path, device="cuda"):
#     kwargs = {"device_map": "auto"}
#     kwargs['load_in_4bit'] = True
#     kwargs['quantization_config'] = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type='nf4'
#     )
#     model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
#     model.to(device)
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#     return model, tokenizer
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
import torch

def load_model(model_path, device="cuda"):
    kwargs = {"device_map": "auto"}
    kwargs['load_in_4bit'] = True
    kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    return model, tokenizer

def get_vision_tower(model):
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda')
    return vision_tower
