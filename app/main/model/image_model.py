import os
import tensorflow as tf
from imageai.Prediction.Custom import ModelTraining, CustomImagePrediction
from app import config
GPU_CONFIG = tf.compat.v1.ConfigProto() 
GPU_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.9
GPU_CONFIG.gpu_options.allow_growth = True

class ImageModel():
    """
    An Image Processing Model. The class can train and predict based on
    the model and region passed
    """
    prediction = None
    model_count = 0

    def __init__(self, region=None, model_name=None, model_type=None):
        self.region = region
        self.model_type = model_type
        self.model_fulldir = os.path.join(config.MODEL_BASEDIR, "{0}/{1}" \
            .format(self.region, self.model_type))
        self.model_name = model_name
        
        tf.keras.backend.clear_session()
        tf.keras.backend.set_session(tf.Session(config=GPU_CONFIG))

    def init_model(self):
        """
        Initializes the image AI model. This also checks the syntax of the request
        if it conforms to the correct folder structure
        """
        self.prediction = CustomImagePrediction()
        self.prediction.setModelTypeAsResNet()

        model_path = os.path.join(self.model_fulldir, "models/{0}".format(self.model_name))
        print("model path")
        print(model_path)
        assert os.path.isfile(model_path), "Model {0} does not exist!".format(self.model_name)

        self.prediction.setModelPath(model_path)

        json_path = os.path.join(self.model_fulldir, "json/model_class.json")
        assert os.path.isfile(json_path), "JSON model class does not exist!"
        self.prediction.setJsonPath(json_path)
        try:
            train_path = os.path.join(self.model_fulldir, "train")
            self.model_count = len(next(os.walk(train_path))[1])
            self.prediction.loadModel(num_objects=self.model_count)
        except:
            raise Exception("Loading the model failed! Ensure that model_count is correct for the chosen model. \
                Retrain if the model does not work and try using the new model")
        self.model_count = self.model_count
        self.prediction = self.prediction

    def predict(self, image):
        """
        Returns the prediction for the particular image.
        """
        ImageModel.init_model(self)
        prediction_list = []
        model_count = self.model_count if self.model_count < 5 else 5
        predictions, probabilities = self.prediction.predictImage(image, result_count=model_count, input_type="array")
        for each_prediction, each_probability in zip(predictions, probabilities):
            prediction_list.append({
                "prediction": each_prediction,
                "probability": each_probability
            })
        return prediction_list
    
    def train(self):
        model_trainer = ModelTraining()
        model_trainer.setModelTypeAsResNet()
        print("TRAIN FULL DIR")
        print(self.model_fulldir)
        model_trainer.setDataDirectory(self.model_fulldir)
        # batch size should be greater than the total image in train or test 
        # https://github.com/OlafenwaMoses/ImageAI/issues/203 
        # https://github.com/h5py/h5py/issues/853
        train_dir = os.path.join(self.model_fulldir, config.TRAIN_FOLDER_NAME)
        print(train_dir)
        #TODO: HANDLE BATCH SIZE TO BE THE NEAREST POWER OF 2 ON LOWEST BETWEEN TRAIN AND TEST
        model_trainer.trainModel(num_objects=len(os.listdir(train_dir)), \
            num_experiments=1000, enhance_data=True, batch_size=16, show_network_summary=True)