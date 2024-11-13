from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载 Qwen 模型
model_name_or_path = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)


# 获取模型结构中的层类名称
def get_layer_classes(model):
    layer_classes = set()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Module,)):
            layer_classes.add(module.__class__.__name__)
    return layer_classes


layer_classes = get_layer_classes(model)
print("Layer Classes:", layer_classes)

"""
❯ python -u "/home/o2igin/Lab/Fun/stanford_alpaca/get_qwen_layer_class_name.py"
Layer Classes: {'Qwen2SdpaAttention', 'Embedding', 'Qwen2RotaryEmbedding', 'Linear', 'Qwen2MLP', 'Qwen2DecoderLayer', 'Qwen2ForCausalLM', 'ModuleList', 'Qwen2RMSNorm', 'SiLU', 'Qwen2Model'}
"""
