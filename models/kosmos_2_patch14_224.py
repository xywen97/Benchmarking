import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

model = AutoModelForVision2Seq.from_pretrained("/data/xiangyu/benchmarkModels/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("/data/xiangyu/benchmarkModels/kosmos-2-patch14-224")

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.route('/generate', methods=['POST'])
def generate_text():
    # Get prompt from the request
    data = request.json
    prompt = data.get("prompt", "")
    image_paths = data.get("image_paths", [])

    print(f"""
        This is the instructblip_flan_t5_xl.py script.
        The prompt is: {prompt}
        The image path is: {image_paths}
    """)

    images = []

    '''
    use a white image as placeholder if we have no images as input
    '''
    if len(image_paths) == 0:
        image_paths.append("/home/xiangyu/project/multimodalEDABenchmarking/models/MiniGPT-4/tmp.png")

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

        return combined_image

    # Combine images and use the combined image for processing
    if len(images) > 1:
        combined_image = combine_images(images)
        # combined_image.save("combined_image.png")
        images = [combined_image]

    inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)

    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=1024,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Specify `cleanup_and_extract=False` in order to see the raw model generation.
    # processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=True)

    # print(processed_text)
    # `<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>.`

    # By default, the generated  text is cleanup and the entities are extracted.
    processed_text, entities = processor.post_process_generation(generated_text)

    print(processed_text)
    # `An image of a snowman warming himself by a fire.`

    # print(entities)
    # `[('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]`
    return jsonify(processed_text)
    
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5015)
