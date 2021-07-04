
from bone_dataset import BoneDataset
from xnet import model as XnetModel

from tensorflow import keras
from sklearn.model_selection import KFold

from tensorflow.keras.optimizers import Adam

from dice import dice_coef, dice_coef_loss
from hausdorf import weighted_hausdorff_distance


def kfold_xnet_test(img_size, all_gen, bone_data, batch_size=8, epochs=2, num_folds=5):
    num_folds = 5
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    metric_names_list = []
    scores_list = []
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
        xnet_model = XnetModel(input_shape=img_size+(3,), classes=1)
        
        xnet_model.compile(optimizer=Adam(learning_rate=1e-3), loss=dice_coef_loss, metrics=[dice_coef, weighted_hausdorff_distance(w=img_size[0], h=img_size[1], alpha=0)])
        
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        
        
        history = xnet_model.fit(train_gen, epochs=epochs, batch_size=batch_size)
        scores = xnet_model.evaluate(test_gen)
        message = f"Score for fold {fold_no}: "
        for metric_name, metric_score in zip(xnet_model.metrics_names, scores):
            message += f"{metric_name} of {metric_score}; "
        print(message)
        # print(f'Score for fold {fold_no}: {xnet_model.metrics_names[0]} of {scores[0]}; {xnet_model.metrics_names[1]} of {scores[1]}')
        scores_list.append(scores)
        metric_names_list.append(xnet_model.metrics_names)

        # Increase fold number
        fold_no = fold_no + 1
    return metric_names_list, scores_list