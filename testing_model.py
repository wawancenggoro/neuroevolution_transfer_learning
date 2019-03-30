PATH_TO_IMAGES = "../images_resized/"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01
NUM_LAYERS = 58
FREEZE_LAYERS = 0
DROP_RATE = 0.0
chromosome = [58, 0, 2, 0]
NUM_OF_EPOCHS = 100

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Scale(224),
        # because scale doesn't always give 224 x 224, this ensures 224 x
        # 224
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# create train/val dataloaders
transformed_datasets = {}
transformed_datasets['train'] = CXR.CXRDataset(
    path_to_images=PATH_TO_IMAGES,
    fold='train',
    transform=data_transforms['train'])
transformed_datasets['val'] = CXR.CXRDataset(
    path_to_images=PATH_TO_IMAGES,
    fold='val',
    transform=data_transforms['val'])

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(
    transformed_datasets['train'],
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0)
dataloaders['val'] = torch.utils.data.DataLoader(
    transformed_datasets['val'],
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0)

# load model
checkpoint_best = torch.load('results/checkpoint')
model = checkpoint_best['model']
epoch_loss = 0.02

# get preds and AUCs on test fold
preds, aucs = E.make_pred_multilabel(
    data_transforms, model, PATH_TO_IMAGES, epoch_loss, CHROMOSOME)