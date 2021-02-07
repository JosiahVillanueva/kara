import io, cv2
import numpy as np

from flask import request, send_file
from flask_restplus import Resource
from PIL import Image

from ..util.alignment import align
from ..util.dto import ImageProcessingDto
from ..util.decorator import upload_parser, multiple_image_parser
from lobe import ImageModel

api = ImageProcessingDto.api
model_parser = ImageProcessingDto.model_parser
upload_parser_with_model = upload_parser(model_parser)
upload_parser = upload_parser()
multiple_image_parser = multiple_image_parser()

def convert_to_cv2_image(werkzeug_image_file):
    in_memory_file = io.BytesIO()
    werkzeug_image_file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    image = cv2.imdecode(data, color_image_flag)
    return image

@api.route("/scan-img")
@api.expect(upload_parser.add_argument("color", type=str, location="form", required=False))
class Realignment(Resource):
    @api.doc("Automatically scans an image")
    def post(self):
        """Automatically scans an image"""
        color = request.form.get("color")
        image = convert_to_cv2_image(request.files["file"])
        documentType = request.form.get("documentType")
        
        aligned_image = align(image, documentType, color)
        
        img = Image.fromarray(aligned_image)
        file_object = io.BytesIO()
        # write PNG in file-object
        img.save(file_object, 'PNG')
        # move to beginning of file so `send_file()` it will read from start    
        file_object.seek(0)

        return send_file(file_object, mimetype='image/PNG')
    

@api.route("/predict")
@api.expect(upload_parser.add_argument("color", type=str, location="form", required=False))
class Predict(Resource):
    @api.doc("Image recognition, recognize uploaded image thru Lobe Tensorflow")
    def post(self):
        image = request.files["file"]
        # Directory of the Tensorflow exported on Lobe
        # model = ImageModel.load('C:\\Project\\kara-master\\kara-master_v2\\kara\\app\\main\\imagerecognition')
        model = ImageModel.load('/var/www/html/kara/app/main/imagerecognition')
        result = model.predict_from_file(image)
        labels = []
    
        for label, confidence in result.labels:
            labels.append([label, confidence])
        
        result = {"Labels": labels, "Prediction": result.prediction}
        
        return result