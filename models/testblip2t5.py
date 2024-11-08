from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests

# model = InstructBlipForConditionalGeneration.from_pretrained("/data/xiangyu/benchmarkModels/instructblip-flan-t5-xl")
# processor = InstructBlipProcessor.from_pretrained("/data/xiangyu/benchmarkModels/instructblip-flan-t5-xl")
# Load model directly
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

processor = AutoProcessor.from_pretrained("/data/xiangyu/benchmarkModels/blip2-flan-t5-xl")
model = AutoModelForVisualQuestionAnswering.from_pretrained("/data/xiangyu/benchmarkModels/blip2-flan-t5-xl")

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model.to(device)

url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
# image = [Image.open("/home/xiangyu/project/multimodalEDABenchmarking/data/tmp/image_0.png").convert("RGB")]
image = [Image.open("/home/xiangyu/project/multimodalEDABenchmarking/models/MiniGPT-4/tmp.png").convert("RGB")]
prompt = """What does the REPAIR_PDN_VIA_LAYER variable in Placement specify?"""
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
