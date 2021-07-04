from keras import backend as K

 
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true) # Extend y_true to one dimension.
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    res_coef = (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
    return res_coef

def dice_coef_loss(y_true, y_pred):
    res_loss = 1. - dice_coef(y_true, y_pred)
    return res_loss