import errno
import io, cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join

from flask import request, send_file
from flask_restplus import Resource
from PIL import Image
from werkzeug.exceptions import BadRequest

from app import config

from ..model.image_model import ImageModel
from ..util.alignment import align
from ..util.dto import ImageProcessingDto
from ..util.decorator import upload_parser, multiple_image_parser

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

@api.route("/manage")
@api.expect(model_parser.remove_argument("model_name"))
class Models(Resource):
    @api.doc("Get all the available models")
    def get(self):
        """Get all the available models"""
        model_type = request.args.get("model_type")
        region = request.args.get("region")
        model_dir = join(config.MODEL_BASEDIR, "{0}/{1}/models/".format(region, model_type))
        try:
            onlyfiles = [f for f in listdir(model_dir) if isfile(join(model_dir, f))]
        except FileNotFoundError:
            return {
                "model_list": []
            }, 200
        return {
            "model_list": onlyfiles
        }, 200

    multiple_image_parser \
        .add_argument("is_train", \
            location='form', \
            type=bool, \
            required=True, \
            help="Location of upload. train folder if true; test folder if false",) \
        .add_argument("category", \
            location='form', \
            type=str, \
            required=True, \
            help="Category of the image. Example: [dog1.jpg, husky.jpg]; category = dog")

    @api.expect(multiple_image_parser)
    def post(self):
        """Uploads images in in preparation for training"""
        model_type = request.args.get("model_type")
        region = request.args.get("region")
        is_train = str(request.form.get("is_train")).lower()
        category = request.form.get("category")
        images = request.files['images']

        if images.content_type is not None:
            model_dir = join(config.MODEL_BASEDIR, "{0}/{1}".format(region, model_type))
            try:
                image_dir = None
                if is_train == "true" or is_train == "none":
                    image_dir = config.TRAIN_FOLDER_NAME
                else:
                    image_dir = config.TEST_FOLDER_NAME
                model_dir = join(model_dir, image_dir, category)
                makedirs(model_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            images = request.files.getlist("images")
            for image in images:
                image.save(join(model_dir, image.filename))
            return {
                "message": "Successfully uploaded {0} images under {1}".format(len(images), image_dir)
            }, 200
        return 400

@api.route("/predict")
@api.expect(upload_parser_with_model)
@api.response(404, "Model not found.")
class Predict(Resource):
    @api.doc("Predicts a model")
    def post(self):
        """Predicts a model"""
        model_name = request.args.get("model_name")
        model_type = request.args.get("model_type")
        region = request.args.get("region")
        image = convert_to_cv2_image(request.files["file"])
        try:
            image_model = ImageModel(region=region, model_name=model_name, model_type=model_type)
            prediction_list = image_model.predict(image=image)
        except Exception as error:
            raise BadRequest(error.args)
        return {
            "prediction_list": prediction_list
        }, 200

@api.route("/train")
@api.expect(model_parser.remove_argument("model_name"))
class Train(Resource):
    @api.doc("Get all the available models")
    def get(self):
        """Get all the available models"""
        model_type = request.args.get("model_type")
        region = request.args.get("region")
        try:
            image_model = ImageModel(region=region, model_type=model_type)
            image_model.train()
        except Exception as error:
            raise BadRequest(error.args)
        return "Training successful!", 200

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
