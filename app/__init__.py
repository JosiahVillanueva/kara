from flask_restplus import Api
from flask import Blueprint

from .main.controller.image_controller import api as image_ns

blueprint = Blueprint("api", __name__)

api = Api(blueprint,
          title="Sunlife ICR/OCR",
          version="1.0",
          description="Initial ICR/OCR APIs"
          )

api.add_namespace(image_ns)