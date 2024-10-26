# Load model directly
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from PIL import Image

local_ckpt_path = "/data/xiangyu/benchmarkModels/blip-image-captioning-large"

# 加载处理器和模型
processor = AutoProcessor.from_pretrained(local_ckpt_path)
model = AutoModelForSeq2SeqLM.from_pretrained(local_ckpt_path)

def generate_caption(image_path):
    # 打开图像
    image = Image.open(image_path)

    # 处理图像
    inputs = processor(images=image, return_tensors="pt")

    # 生成图像描述
    outputs = model.generate(**inputs)

    # 解码生成的描述
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# 示例用法
image_path = "/home/xiangyu/project/multimodalEDABenchmarking/models/MiniGPT-4/examples/test.png"
caption = generate_caption(image_path)
print("生成的图像描述:", caption)