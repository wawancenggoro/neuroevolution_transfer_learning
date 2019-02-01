import cxr_dataset as CXR
import eval_model as E
import model as M


# you will need to customize PATH_TO_IMAGES to where you have uncompressed
# NIH images
PATH_TO_IMAGES = "../images_resized/"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.1
NUM_LAYERS = 7
FREEZE_LAYERS = 4
DROP_RATE = 0.2
chromosome = [7, 4, 1, 2]
NUM_OF_EPOCHS = 100
preds, aucs = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY, NUM_LAYERS, FREEZE_LAYERS, DROP_RATE, chromosome, NUM_OF_EPOCHS)

