import matplotlib.pyplot as plt
from tensorflow_core.python.keras.callbacks import ModelCheckpoint

from csvReader import csvReader
from model import *
from data import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from myGenerator import load_train, load_test

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
# myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
# myGene = trainGenerator(2,r'E:\训练集\carChallenge','train_hq','train_masks',data_gen_args,save_to_dir = None)
# myGene = csvReader.trainGenerator(batch_size=2, train_path=r'E:\训练集\carChallenge\train_hq',
#                                   label_path=r"E:\训练集\carChallenge\train_masks", aug_dict=data_gen_args)
# myGene = csvReader.generator_with_csv(r"E:\训练集\carChallenge\mycsv.csv")
# myGene = csvReader.generator_with_csv(r"mycsv.csv")
# myGene = load_train(r"E:\train_data\carChallenge\mycsv.csv", 512, 512)

# aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
# 	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
# 	horizontal_flip=True, fill_mode="nearest")


model = unet()
# model_checkpoint_loss = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True,
#                                         save_weights_only=True, mode='auto', period=1)
# model_checkpoint_acc = ModelCheckpoint('unet_membrane_acc.hdf5', monitor='accuracy', verbose=1, save_best_only=True,
#                                        save_weights_only=True, mode='auto', period=1)
# model_checkpoint_val_loss = ModelCheckpoint('unet_membrane_acc.hdf5', monitor='val_loss', verbose=1,
#                                             save_best_only=True,
#                                             save_weights_only=True, mode='auto', period=1)
# model_checkpoint_val_acc = ModelCheckpoint('unet_membrane_acc.hdf5', monitor='val_acc', verbose=1, save_best_only=True,
#                                            save_weights_only=True, mode='auto', period=1)

history = model.fit_generator(load_train(r"/home/ubuntu/wzrData/cartest/mycsvOnlinux.csv", 512, 512, 4), workers=1,
                              steps_per_epoch=1197, epochs=50, validation_data=load_test("/home/ubuntu/wzrData/cartest/test_dataOnlinux.csv", 512, 512, 300)
                              )

# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene, 30, verbose=1)
# saveResult("data/membrane/test", results)

model.save('modelWithWeight.h5')
model.save_weights('fine_tune_model_weight')
# print(results)
print(history.history)
# print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
