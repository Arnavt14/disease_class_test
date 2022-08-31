# All the functions will run here including the streamlit front end
from models import edema_classification
from constants import Edema_constants

# for training the models


edema_classification.load_data()
edema_classification.scaling_data(Edema_constants.data)
edema_classification.test_train_split()
edema_classification.create_model()
edema_classification.train_model()
edema_classification.eval()
edema_classification.save_model()

