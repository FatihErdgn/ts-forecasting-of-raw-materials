import sys
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback,EarlyStopping
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import threading
import random

# Data to train model
data = pd.read_csv('hourly_wages.csv')


X = data.drop('wage_per_hour', axis=1)
y = data['wage_per_hour']
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=42) #random state for training the data with the same random values for validation and training
                                                                                        #test size for validation data batch. uses 0.8 of the data for training    
# Model Construction
n_cols = X_train.shape[1]
early_stopping_monitor = EarlyStopping(patience=20) #should beat the best score after 20 epochs

model = Sequential()
model.add(Dense(150, activation='relu', input_shape=(n_cols,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1))

# Compiling the model
lr = 0.01
sgd_optimizer = SGD(lr=lr)
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

# loss value lists
losses = []
val_losses = []

# Callback class
class PlotLosses(Callback):
    def on_epoch_end(self, epoch, logs={}):
        losses.append(logs.get('loss'))
        val_losses.append(logs.get('val_loss'))
        window.update_figure()

# Plot Callback
plot_losses = PlotLosses()

# PyQt5 App Window
class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Training and Validation Loss'
        self.width = 1024
        self.height = 768
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.canvas = PlotCanvas(self, width=5, height=4)
        self.canvas.move(0, 0)
        
        # Button for using the make_prediction function
        self.predict_button = QPushButton('Predict Wage', self)
        self.predict_button.move(850, 650)  # Düğmenin konumunu ayarla
        self.predict_button.clicked.connect(self.make_prediction)  # Düğmeye basıldığında make_prediction fonksiyonunu çalıştır

        self.show()
        
        
    def update_figure(self):
        self.canvas.plot()
    
    def make_prediction(self):
       new_data = [[0,12,23,41,0,1,0,0,0]]  # sample data for testing (real value of this data: 9.56,)

       # Predict Wage for Testing
       predicted_wage = model.predict(new_data)

       print(f"Tahmini saatlik ücret (Örnek Veri): {predicted_wage[0][0]}")

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=10, height=8, dpi=180):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def plot(self):
        self.axes.clear()
        self.axes.plot(losses, label='Training loss')
        self.axes.plot(val_losses, label='Validation loss')
        self.axes.legend(loc='upper right')
        self.axes.scatter(len(val_losses)-1, val_losses[-1], color='red',s=5)  # Adding a red marker at the last point
        # Formatting the text label to show only three decimal places
        self.axes.text(len(val_losses)-1, val_losses[-1], f' {val_losses[-1]:.3f}', verticalalignment='center')
        self.axes.scatter(len(losses)-1, losses[-1], color='blue',s=5)  # Adding a red marker at the last point
        # Formatting the text label to show only three decimal places
        self.axes.text(len(losses)-1, losses[-1], f' {losses[-1]:.3f}', verticalalignment='bottom')
        self.draw()

# Model Training Function
def train_model():
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), callbacks=[plot_losses,early_stopping_monitor], epochs=500)
    #model.save('model.h5')
    
    #For using the model
    #loaded_model = load_model('model.h5')
    #loaded_model.predict(pred_data)
    #loaded_model.summary() to verify the model structure
    
# PyQt5 App
app = QApplication(sys.argv)
window = App()

# Train the model
training_thread = threading.Thread(target=train_model)
training_thread.start()

sys.exit(app.exec_())



