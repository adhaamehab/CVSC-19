from flask import Flask, send_file, request
from io import BytesIO
import torchvision
import PIL
import base64
import numpy as np
from segmenetation import instance_segmentation_api

app = Flask(__name__)


@app.route('/', methods=['GET'])
def health_endpoint():
    return 'Ok!'


@app.route('/segmentation', methods=['POST', 'GET'])
def seg_endpoint():
    file = request.get_json()['filePath']
    print(file)
    return instance_segmentation_api(model, file)


if __name__ == "__main__":
    model = model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True, progress=True)

    model.eval()

    app.run(host='0.0.0.0', port=5019, debug=True)
