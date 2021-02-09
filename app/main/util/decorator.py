import werkzeug
from flask_restplus import reqparse

def upload_parser(root_parser=None):
    print(root_parser)
    if root_parser is None:
        file_upload = reqparse.RequestParser()
    else:
        file_upload = root_parser.copy()

    file_upload.add_argument("file", \
        type=werkzeug.datastructures.FileStorage, \
        location="files", \
        required=True, \
        help="File upload. Supported files: image.")
    return file_upload

def upload_predict(root_parser=None):
    if root_parser is None:
        file_upload = reqparse.RequestParser()
    else:
        file_upload = root_parser.copy()

    file_upload.add_argument("file", \
        type=werkzeug.datastructures.FileStorage, \
        location="files", \
        required=True, \
        help="File upload. Supported files: image.")
    return file_upload

def multiple_image_parser(root_parser=None):
    if root_parser is None:
        file_upload = reqparse.RequestParser()
    else:
        file_upload = root_parser.copy()

    file_upload \
        .add_argument("images", \
            type=werkzeug.datastructures.FileStorage, \
            location="files", \
            required=True, \
            help="File upload. Supported files: image.", \
            action="append")
    return file_upload
