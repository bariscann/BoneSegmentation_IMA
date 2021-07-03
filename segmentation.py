from bone_data import BoneData
from bone_dataset import BoneDataset
from xnet import model as XnetModel
from tensorflow import keras
from sklearn.model_selection import KFold
from scipy.spatial.distance import directed_hausdorff, dice
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import numpy as np


smooth = 1. # Used to prevent the denominator from 0.
def dice_coef(y_true, y_pred):
    # y_true.astype(np.float32)
    print(y_pred.dtype, y_true.dtype)
    print(y_pred.shape, y_true.shape)
    
    # y_true = y_pred
    # y_pred = y_true
    y_true_f = K.flatten(y_true) # Extend y_true to one dimension.
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

#The functions return our metric and loss
bone_data = BoneData()
img_size = (160, 160)
num_classes = 2
batch_size = 8
epochs = 2 # to test segmentation immediatly
all_gen = BoneDataset(batch_size=1,
                      img_size=img_size,
                      input_img_paths=bone_data.all_data_files,
                      target_img_paths=bone_data.all_label_files)


num_folds = 5
# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
acc_per_fold = []
loss_per_fold = []
for train, test in kfold.split(all_gen):
    train_gen = BoneDataset(batch_size=batch_size,
                      img_size=img_size,
                      input_img_paths=bone_data.all_data_files[train],
                      target_img_paths=bone_data.all_label_files[train])
    test_gen = BoneDataset(batch_size=batch_size,
                      img_size=img_size,
                      input_img_paths=bone_data.all_data_files[test],
                      target_img_paths=bone_data.all_label_files[test]) 
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()
    xnet_model = XnetModel(input_shape=(160, 160, 3), classes=2)
    # jaccard_distance kullanmis diger proje loss function olarak
    # optimizer = Adam(lr=1e-4)
    # xnet_model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=[dice, directed_hausdorff])
    xnet_model.compile(optimizer=Adam(learning_rate=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
    # xnet_model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    
    history = xnet_model.fit(train_gen, epochs=epochs)
    scores = xnet_model.evaluate(test_gen)
    
    print(f'Score for fold {fold_no}: {xnet_model.metrics_names[0]} of {scores[0]}; {xnet_model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1