from flask import Flask, request, jsonify
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model_path = "/data/xiangyu/benchmarkModels/CHENCHEN/huggingface/hub/models--internlm--internlm-xcomposer-vl-7b/snapshots/8a8a3ae062068c45a0c25875146237cc8b5e20e1"
# model_path = "/data/xiangyu/benchmarkModels/CHENCHEN/huggingface/hub/models--internlm--internlm2_5-7b-chat/snapshots/9b8d9553846ecf6393f3408fa9d3ec9928fdab4d"
model_path = "/data/xiangyu/benchmarkModels/CHENCHEN/huggingface/hub/models--internlm--internlm-xcomposer2-vl-7b/snapshots/c67bd06390dbe068a582c6561570725b1289a7c5"

model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.tokenizer = tokenizer

@app.route('/generate', methods=['POST'])
def generate_text():
    # Get prompt from the request
    data = request.json
    prompt = data.get("prompt", "")
    image_paths = data.get("image_paths", [])

    images = []

    '''
    use a white image as placeholder if we have no images as input
    '''
    # if len(image_paths) == 0:
    #     image_paths.append("/home/xiangyu/project/multimodalEDABenchmarking/models/MiniGPT-4/tmp.png")

    for i in range(len(image_paths)):
        image = Image.open(image_paths[i]).convert("RGB")
        # 获取图像的宽度和高度
        width, height = image.size
        # 计算新的尺寸，保持长边为512
        if width > height:
            new_width = 1024
            new_height = int((1024 / width) * height)
        else:
            new_height = 1024
            new_width = int((1024 / height) * width)

        # 调整图像大小
        resized_image = image.resize((new_width, new_height))

        images.append(resized_image)

    def combine_images(images):
        widths, heights = zip(*(i.size for i in images))

        max_width = max(widths)
        total_height = sum(heights)

        # Calculate the number of rows needed
        num_images = len(images)
        num_rows = (num_images + 1) // 2  # Each row has 2 images

        # Create a new image with the appropriate size
        combined_image = Image.new('RGB', (max_width * 2, max(heights) * num_rows))

        y_offset = 0
        for i in range(0, num_images, 2):
            x_offset = 0
            for j in range(2):
                if i + j < num_images:
                    combined_image.paste(images[i + j], (x_offset, y_offset))
                    x_offset += max_width
            y_offset += max(heights)

        # combined_image = combined_image.resize((512, 512))

        return combined_image

    # Combine images and use the combined image for processing
    # Qwen vl do not support images numbers > 1
    # it reuqires the image path as input, not the loaded images
    if len(images) > 1:
        combined_image = combine_images(images)
        combined_image_name = "combined_image_internlm_xcomposer2_vl.png"
        combined_image.save(combined_image_name)
        combined_image_path = f"/home/xiangyu/project/multimodalEDABenchmarking/models/{combined_image_name}"
        images = [combined_image_path]
    else:
        images = image_paths

    print(f"""
        This is the Llama-3___2-90B-Vision-Instruct.py script.
        The prompt is: {prompt}
        The image path is: {images}
    """)
    # with torch.cuda.amp.autocast():
    #     response, _ = model.chat(tokenizer, query=query, image=image_paths, history=[], do_sample=False)
    if len(images) > 0:
        imagesPrompt = []
        for i in range(len(images)):
            imagesPrompt.append(f"<ImageHere>; ")
        
        image_prompt = ''.join(imagesPrompt) 
        prompt = image_prompt + " " + prompt

        # response = model.generate(prompt, images[0])
        with torch.cuda.amp.autocast():
            response, _ = model.chat(tokenizer, query=prompt, image=images[0], history=[], do_sample=False)
    else:
        with torch.cuda.amp.autocast():
            response, _ = model.chat(tokenizer, query=prompt, image=None, history=[], do_sample=False)
    print(response)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5031)
