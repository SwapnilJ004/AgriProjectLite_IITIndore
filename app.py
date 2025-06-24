import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

app = Flask(__name__)

class_labels = ['Healthy', 'Not Healthy']
class_preventive_measures = ["Your crop is healthy", "Please consult the local agriculture officers"]

img_rows, img_cols = 224, 224
image_size = [224, 224, 3]

# Load TFLite model globally
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check(path):
    img = load_img(path, target_size=image_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0).astype('float32') / 255.0

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(output)
    accuracy = int(np.max(output) * 100)
    return [index, accuracy]

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            try:
                file_extension = file.filename.split('.')[-1]
                file_path = os.path.join('static/images', "testing-image." + file_extension)
                file.save(file_path)

                result = check(file_path)

                if int(result[1]) >= 50:
                    disease_name = class_labels[0]
                    accuracy = result[1]
                    preventive_measures = class_preventive_measures[0]
                else:
                    disease_name = class_labels[1]
                    accuracy = 100 - int(result[1])
                    preventive_measures = class_preventive_measures[1]

                return render_template('predict.html',
                                       disease_name=disease_name,
                                       user_image=file_path,
                                       accuracy=accuracy,
                                       preventive_measures=preventive_measures)
            except Exception as e:
                return f"Error: {str(e)}"
        else:
            return redirect(url_for('predict'))

    return redirect(url_for('predict'))

@app.route('/download-image/<path:filename>')
def download(filename):
    return send_from_directory('static', filename, as_attachment=True,
                               mimetype='image/jpg',
                               download_name=(str(filename) + '.jpg'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
