import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Modeli yükle
model = load_model('C:/DL Python Project/mnist_model.h5')

def visualize_image(img, title="Image"):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

# Çizim için canvas oluştur
def create_canvas():
    canvas = tk.Canvas(window, width=280, height=280, bg='white')
    canvas.pack()

    # Çizim yapma fonksiyonu
    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill='black', width=12)
        draw.line([x1, y1, x2, y2], fill='black', width=12)

    canvas.bind('<B1-Motion>', paint)
    return canvas

# Rakamı çevreleyen kutuyu bul
def get_bounding_box(image):
    img_array = np.array(image)
    rows = np.any(img_array, axis=1)
    cols = np.any(img_array, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax

# Resmi yeniden boyutlandırma ve padding ekleme
def resize_with_padding(img, size, min_padding=3):
    width, height = img.size

    # Yeniden boyutlandırılacak maksimum boyutları hesaplama (min_padding çıkarılır)
    max_new_width = size - (min_padding * 2)
    max_new_height = size - (min_padding * 2)

    # Oranı koruyarak yeniden boyutlandırma
    scale = min(max_new_width / width, max_new_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Padding ekleme - Yatay ve Dikey olarak
    new_img = Image.new('L', (size, size), 0)
    x_padding = max((size - new_width) // 2, min_padding)
    y_padding = max((size - new_height) // 2, min_padding)
    new_img.paste(img, (x_padding, y_padding))
    return new_img

# Resmi işle ve modele tahmin ettir
def predict_digit():
    image.save('digit.png')
    img = Image.open('digit.png').convert('L')
    visualize_image(img, "Original Grayscale Image")

    img = ImageOps.invert(img)
    visualize_image(img, "Inverted Image")

    # xmin, ymin, xmax, ymax = get_bounding_box(img)
    # img = img.crop((xmin, ymin, xmax, ymax))
    # visualize_image(img, "Cropped Image")
    
    img = resize_with_padding(img, 28)
    visualize_image(img, "Resized with Padding")

    img_array = np.array(img)
    img_array = img_array.reshape(1, 784)
    img_array = img_array / 255.0

    probabilities = model.predict(img_array)[0]
    predicted_digit = np.argmax(probabilities)
    label_predict.config(text='Predicted Digit: ' + str(predicted_digit))

    # Tahmin olasılıklarını görselleştir
    bars = plt.bar(range(10), probabilities)
    plt.xlabel('Digits')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.xticks(range(10))
    for bar in bars:
        yval = bar.get_height()
        if yval > 0.005:
            plt.text(bar.get_x() + bar.get_width()/2, yval, '%' + str(round(yval*100, 2)), ha='center', va='bottom')

    plt.show()

# Çizimi temizle
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill='white')

# GUI Penceresi
window = tk.Tk()
window.title('Digit Recognizer')

image = Image.new('RGB', (280, 280), 'white')
draw = ImageDraw.Draw(image)
canvas = create_canvas()

label_predict = tk.Label(window, text='Predicted Digit: None', font=('Helvetica', 12))
label_predict.pack()

btn_predict = tk.Button(window, text='Predict Digit', command=predict_digit, bg='green')
btn_predict.pack()

btn_clear = tk.Button(window, text='Clear Canvas', command=clear_canvas, bg='red')
btn_clear.pack()

window.mainloop()
