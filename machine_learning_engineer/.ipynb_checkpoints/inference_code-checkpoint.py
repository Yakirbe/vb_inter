import numpy as np
from tensorflow import keras

def inference_image(im_fn, model = None):
    if model is None:
        model = keras.applications.resnet50.ResNet50(weights='imagenet')
    img = keras.preprocessing.image.load_img(im_fn, target_size=(224, 224))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.resnet50.preprocess_input(x)
    preds = model.predict(x)
    dpred = keras.applications.resnet50.decode_predictions(preds, top=3)[0]
    print('Predicted:', dpred)
    return dpred