import tensorflow as tf
tf.__version__
#tf.test.is_gpu_available()
print("*****************Found GPU: {}***********************".format(str(tf.config.list_physical_devices('GPU'))))
from tensorflow import keras
from tensorflow.keras import backend as K
import os
import numpy as np
import sys
sys.path.append('/home/sam/work/template-ocr')
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import cv2
from tensorflow.keras.preprocessing.image import NumpyArrayIterator


base_path = '/home/sam/work/template-ocr'
data_schema_path = os.path.join(base_path, "data", "words.txt")
data_path = os.path.join(base_path, "data", "words")
char_list_path = os.path.join(base_path, "data", "charList.txt")
model_dir = os.path.join(base_path, "model")
checkpoint_dir = os.path.join(model_dir, "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "cp.ckpt")

maxTextLen = 32
imgSize = (32, 128)
batch_size = 50
numTrainSamplesPerEpoch = 80000
numOfEpochs = 2
pool = 2

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
#['inputs', 'the_labels', 'input_length', 'label_length']
class CTCImageDataGenerator(keras.preprocessing.image.ImageDataGenerator):
    
    def __init__(self, rescale=1./255, shear_range=0.2, zoom_range=0.2, rotation_range=20, **kwargs):
        super().__init__(self, rescale, shear_range, zoom_range, rotation_range, **kwargs)
        
    def _labels(self, y, maxTextLen, labelencoder):
        y = [sample[0] for sample in y]
        _new_labels = np.full((len(y), maxTextLen), labelencoder.transform([' '])[0])
        _input_len = np.full((len(y), 1), maxTextLen-2)
        _label_len = np.zeros((len(y), 1))
        #_label_len = np.full((len(y), 1), maxTextLen-2)
        for i, word in enumerate(list(y)):
            try:
                _temp_array_first = labelencoder.transform(list(word)).transpose()
                _new_labels[i][:min(len(_temp_array_first), maxTextLen)] = _temp_array_first
                _label_len[i] = len(word) if len(word) > 0 else 1
            except:
                print("Error at row no. {}".format(str(i)))
        return [_new_labels, _input_len, _label_len]
    
    def flow_CTC(self, x, current_samples, labelencoder, char_list, maxTextLen, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='',
             save_format='png', subset=None):
        ctc_params = self._labels(current_samples, maxTextLen, labelencoder) 
        """inputs = {'inputs': x,
                  'input_length': input_len,
                  'the_labels': y_label,
                  'label_length': label_len,}
        """
        inputs_CTC = [x, ctc_params]
        return NumpyArrayIterator(
                    inputs_CTC,
                    y,
                    self,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    sample_weight=sample_weight,
                    seed=seed,
                    data_format=self.data_format,
                    save_to_dir=save_to_dir,
                    save_prefix=save_prefix,
                    save_format=save_format,
                    subset=subset,
                    dtype=self.dtype
                )
        
        

with open(char_list_path) as cp:
    char_list = []
    for line in cp:
        __temp_list = [char_list.append(char) for char in list(line) if char not in char_list and char != '\n']
    char_list = list(set(char_list))
    
with open(data_schema_path) as fp:
    chars = set()
    bad_samples = []
    bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
    samples = []
    for line in fp:
        if not line or line[0]=='#':
            continue
        
        lineSplit = line.strip().split(' ')
        assert len(lineSplit) >= 9
        # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
        fileNameSplit = lineSplit[0].split('-')
        fileName = data_path + '/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'
        if lineSplit[0] + '.png' in bad_samples_reference:
            continue
 		# GT text are columns starting at 9
        gtText = ' '.join(lineSplit[8:maxTextLen])
        chars = chars.union(set(list(gtText)))
        
        # check if image is not empty
        if not os.path.getsize(fileName):
            bad_samples.append(lineSplit[0] + '.png')
            print("Damaged image found..")
            bad_samples_reference.append(lineSplit[0] + '.png')
        else:
        # put sample into list
            samples.append([gtText, fileName])

# split into training and validation set: 95% - 5%
splitIdx = int(0.80 * len(samples))
trainSamples = samples[:splitIdx]
validationSamples = samples[splitIdx:]

dataAugmentation = True
currIdx = 0
random.shuffle(trainSamples)
current_samples = trainSamples[:numTrainSamplesPerEpoch]

# list of all chars found in dataset
charList = sorted(list(chars))
labelencoder = LabelEncoder().fit(char_list)
transformed_labels = labelencoder.transform(char_list)

ar = np.array([[[10,20],[20,10]],[[30,40],[40,30]]])
ar = np.delete(ar,0,0)

#i = 201
#y = validationSamples[i][0]
def map_labels(current_samples, X, labelencoder, maxTextLen):
    y_orig = [sample[0] for sample in current_samples]
    new_labels = np.full((len(y_orig), maxTextLen, len(char_list)), labelencoder.transform([' '])[0])
    error_index = []
    for i, y in enumerate(list(y_orig)):
        try:
            _temp_array_first = labelencoder.transform(list(y)).transpose()
            temp_array = keras.utils.to_categorical(_temp_array_first, num_classes=len(char_list))
            new_labels[i][:min(len(temp_array), maxTextLen)] = temp_array
        except:
            print(f"Error at row no. {str(i)}. Length of current_samples is {str(len(current_samples))} ")
            error_index.append(i)
    #new_labels = new_labels.reshape((new_labels.shape[0], new_labels.shape[1], 1))
    for ind in error_index:
        del current_samples[ind]
        del y_orig[ind]
        X = np.delete(X, ind, 0)
        new_labels = np.delete(new_labels, ind, 0)
    return current_samples, X, y_orig, new_labels
del current_samples[[1,10,15]]

def inverse_label_map(y_array, labelencode, maxTextLen):
    text_list = []
    y_array = y_array.argmax(len(y_array.shape)-1)
    for y_test in list(y_array):
        text = ''.join(labelencoder.inverse_transform([int(val) for val in list(y_test)]))
        text_list.append(text)
    return y_array, text_list
      
    

def create_input_format(current_samples, imgSize):
    image_batch_list  = []
    current_samples_path = [sample[1] for sample in current_samples]
    # Read images from disk
    i = 0
    for sample_path in current_samples_path:
        i+=1
        total = len(current_samples_path)
        if i%5000==0:
            print("Completed {} rows out of total {}".format(str(i), str(total)))
        image = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (imgSize[1], imgSize[0]))
        image = image.reshape((image.shape[0],image.shape[1],1))
        image_batch_list.append(image)
    return np.array(image_batch_list)


def CreateImageDataGenerator(X, rescale=1./255, shear_range=0.2, zoom_range=0.2, rotation_range=20, **kwargs):
    obj = CTCImageDataGenerator(rescale=rescale, shear_range=shear_range, zoom_range=zoom_range, rotation_range=rotation_range, **kwargs)
    obj.fit(X)
    return obj


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

#Setting train data
X_train = create_input_format(current_samples, imgSize)
current_samples, X_train, y_train_true, y_train = map_labels(current_samples, X_train, labelencoder, maxTextLen)
Img_obj = CreateImageDataGenerator(X_train)

#seeting validation data
X_val = create_input_format(validationSamples, imgSize)
validationSamples, X_val, y_val_true, y_val = map_labels(validationSamples, X_val, labelencoder, maxTextLen)
y_val_true_encoded, y_val_true = inverse_label_map(y_val, labelencoder, maxTextLen)
Img_obj_val = CreateImageDataGenerator(X_val, shear_range=0, zoom_range=0, rotation_range=0)
#y_val_true[3253][0]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Flatten, LSTM, Input, Permute, Reshape, Conv2D, MaxPooling2D, Lambda, TimeDistributed, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, SGD

#model = Sequential()
inputs = Input(shape=(imgSize[0], imgSize[1], 1), name="inputs")

########### CNN
model_conv_1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='model_conv_1')(inputs)
model_pool_1 = MaxPooling2D(pool_size=(pool, pool), name='model_pool_1')(model_conv_1) 
model_drop_1 = Dropout(0.2, name='model_drop_1')(model_pool_1) #Output (h/2, w/2) - 16

model_conv_2 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='model_conv_2')(model_drop_1)
model_pool_2 = MaxPooling2D(pool_size=(pool, pool), name='model_pool_2')(model_conv_2) 
model_drop_2 = Dropout(0.2, name='model_drop_2')(model_pool_2) #Output (h/2, w/2) - 8

model_conv_3 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='model_conv_3')(model_drop_2)
model_drop_3 = Dropout(0.2, name='model_drop_3')(model_conv_3) #Output (h, w) - 8

model_conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='model_conv_4')(model_drop_3)
bnorm1 = BatchNormalization(name='bnorm1')(model_conv_4)
model_pool_3 = MaxPooling2D(pool_size=(pool, pool-1), name='model_pool_3')(bnorm1)
model_drop_4 = Dropout(0.2, name='model_drop_4')(model_pool_3) #Output (h/2, w/2) - 4 


model_conv_5 = Conv2D(256, kernel_size=(1, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='model_conv_5')(model_drop_4)
model_drop_5 = Dropout(0.2, name='model_drop_5')(model_conv_5) #Output (h, w) - 4

model_conv_6 = Conv2D(512, kernel_size=(1, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='model_conv_6')(model_drop_5)
model_pool_4 = MaxPooling2D(pool_size=(pool, pool-1), name='model_pool_4')(model_conv_6)
model_drop_6 = Dropout(0.2, name='model_drop_6')(model_pool_4) #Output (h, w) - 2

model_conv_7 = Conv2D(512, kernel_size=(1, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='model_conv_7')(model_drop_6)
bnorm2 = BatchNormalization(name='bnorm2')(model_conv_7)
model_pool_5 = MaxPooling2D(pool_size=(pool, pool-1), name='model_pool_5')(bnorm2)
model_drop_7 = Dropout(0.2, name='model_drop_7')(model_pool_5) #Output (h, w) - 1

#Squeezing to fit in LSTM
squeeze_1 = Lambda(lambda x: tf.keras.backend.squeeze(x, axis=1))(model_drop_7) #Removing height component and only keeping width

#Dense Layer
dense_1 = Dense(maxTextLen, activation='relu', name='dense_1', kernel_initializer='he_normal')(squeeze_1)

#LSTM
model_LSTM_1 = Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout = 0.2, return_sequences=True, kernel_initializer='he_normal', name="lstm_1"))(dense_1)
bnorm3 = BatchNormalization(name='bnorm3')(model_LSTM_1)

model_LSTM_2 = Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout = 0.2, return_sequences=True, kernel_initializer='he_normal', name="lstm_2"))(bnorm3)
bnorm4 = BatchNormalization(name='bnorm4')(model_LSTM_2)

model_LSTM_3 = Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout = 0.2, return_sequences=True, kernel_initializer='he_normal', name="lstm_3"))(bnorm4)
bnorm5 = BatchNormalization(name='bnorm5')(model_LSTM_3)

#Dense
dense_2 = Dense(len(char_list)+1, kernel_initializer='he_normal', name="dense_2")(bnorm5)
y_pred = Activation('softmax', name='activation')(dense_2)
#y_pred = Lambda(lambda x: tf.transpose(x, (1,0,2)), name="transpose")(y_pred)

labels = Input(name='the_labels', shape=[maxTextLen], dtype='float32') # (None ,32)
input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)
ada = Adadelta()
sgd = SGD(learning_rate=0.02,
              decay=1e-6,
              momentum=0.9,
              nesterov=True)
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
model.summary()
model.compile(loss={'ctc': lambda labels, y_pred: y_pred}, optimizer=ada)
model.fit_generator(Img_obj.flow_CTC(X_train, current_samples, labelencoder, char_list, maxTextLen, y_train, batch_size=100), epochs=2, validation_data=Img_obj_val.flow_CTC(X_val, validationSamples, labelencoder, char_list, maxTextLen, y_val, batch_size=100), callbacks=[cp_callback])


############################################ RESEARCH
model = Model(inputs=inputs, outputs=y_pred)
model.summary()
model.compile(optimizer = keras.optimizers.RMSprop(learning_rate=0.001), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#model.fit(np.array(image_batch_list), y_train)
model.fit_generator(Img_obj.flow_CTC(X_train, y_train, batch_size=50), epochs=10, validation_data=Img_obj_val.flow(X_val, y_val, batch_size=50), callbacks=[cp_callback])
model.save(os.path.join(model_dir, 'HW_OCR_CRNN_No_CTC.h5'))
#Prediction from trained model
y_val_pred = model.predict_generator(Img_obj_val.flow(X_val, y_val, batch_size=50))
y_val_pred_inversed = inverse_label_map(y_val_pred, labelencoder, maxTextLen)
y_val_true = inverse_label_map(y_val, labelencoder, maxTextLen)

for i in range(5):
    i_rand = random.randint(0, len(y_val_true))
    print([y_val_pred_inversed[i_rand],  y_val_true [i_rand]])


test_samples = trainSamples[numTrainSamplesPerEpoch:]
X_test = create_input_format(test_samples, imgSize)
y_test = map_labels(test_samples, labelencoder, maxTextLen)
Img_obj_test = CreateImageDataGenerator(X_test, shear_range=0, zoom_range=0, rotation_range=0)

y_test_pred = model.predict_generator(Img_obj_test.flow(X_test, y_test, batch_size=50))
y_test_pred_inversed = inverse_label_map(y_test_pred, labelencoder, maxTextLen)

_temp_array = Img_obj.flow_CTC(X_train, current_samples, labelencoder, char_list, maxTextLen, y_train, batch_size=2050)
temp_array_ = _temp_array[0][0]
temp_array_[1][2039]
labelencoder.inverse_transform(temp_array_[1][2039])