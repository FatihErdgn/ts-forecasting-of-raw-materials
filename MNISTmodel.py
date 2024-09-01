from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# MNIST veri setini yükle
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Girdi verilerini 0-1 aralığında normalize et
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255

# Etiketleri one-hot encoded formata dönüştür
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Eğitim verilerini daha da bölmek için train_test_split kullan
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


model = Sequential()

# Add the first hidden layer
model.add(Dense(100,activation='sigmoid',input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(100,activation='sigmoid'))

# Add the output layer
model.add(Dense(10,activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
model.save('mnist_model.h5')