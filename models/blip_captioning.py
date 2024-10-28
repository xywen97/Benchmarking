from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

app = Flask(__name__)

# 加载处理器和模型
local_ckpt_path = "/data/xiangyu/benchmarkModels/blip-image-captioning-large"
pipe = pipeline("image-to-text", model=local_ckpt_path)

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    # 从请求中获取图像路径
    data = request.json
    image_path = data.get("image_path", "")
    image_path = f"/home/xiangyu/project/multimodalEDABenchmarking/{image_path}"

    print(f"""
        This is the blip_captioning.py script.
        The image path is: {image_path}
    """)

    # 打开图像
    image = Image.open(image_path)

    # 使用pipeline生成图像描述
    caption = pipe(image)[0]['generated_text']
    image.close()
    return jsonify({"caption": caption})

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5010)