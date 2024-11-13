import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
# model_path = "/data/xiangyu/benchmarkModels/CHENCHEN/huggingface/hub/models--internlm--internlm-xcomposer-vl-7b/snapshots/8a8a3ae062068c45a0c25875146237cc8b5e20e1"
model_path = "/data/xiangyu/benchmarkModels/internlm-xcomposer2-vl-7b"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model.tokenizer = tokenizer

# support only one image
query = '<ImageHere><ImageHere>Please describe this image in detail.'
image = ['aiyinsitan.jpg', 'combined_image_hpt_1_5_edge.png']
with torch.cuda.amp.autocast():
  response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
print(response)
#The image features a quote by Oscar Wilde, "Live life with no excuses, travel with no regret,"
# set against a backdrop of a breathtaking sunset. The sky is painted in hues of pink and orange,
# creating a serene atmosphere. Two silhouetted figures stand on a cliff, overlooking the horizon.
# They appear to be hiking or exploring, embodying the essence of the quote.
# The overall scene conveys a sense of adventure and freedom, encouraging viewers to embrace life without hesitation or regrets.
