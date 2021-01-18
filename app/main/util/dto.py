from flask_restplus import Namespace, fields, reqparse
import copy
import werkzeug

class ImageProcessingDto:
    api = Namespace("image-processing", description="Image processing related modules")

    model_parser = api.parser()
    model_parser.add_argument("model_name", type=str, location="args", required=True)
    model_parser.add_argument("model_type", type=str, location="args", required=True)
    model_parser.add_argument("region", type=str, location="args", required=True)