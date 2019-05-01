import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from preprocess import prepare_model_input_data


def build_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    return model

model = build_model()

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model.
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# Write a unique file name that includes the epoch and the validation acc per epoch.
filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"

checkpoint = ModelCheckpoint(
    "./models/{}.model".format(
        filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
)

train_x, train_y, validation_x, validation_y = prepare_model_input_data()

# Train model.
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)

# Score model.
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model.
model.save("./models/{}".format(NAME))