import cxr_dataset as CXR
import eval_model as E
import model as M


# you will need to customize PATH_TO_IMAGES to where you have uncompressed
# NIH images
PATH_TO_IMAGES = "../sample_images/"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01
NUM_LAYERS = 58
FREEZE_LAYERS = 58
DROP_RATE = 0.0
preds, aucs = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY, NUM_LAYERS, FREEZE_LAYERS, DROP_RATE)

