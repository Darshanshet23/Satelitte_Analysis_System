# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and class dictionary
model = load_model('model/unet_epoch_20.h5')
class_df = pd.read_csv('model/class_dict.csv')
colormap = class_df[['r', 'g', 'b']].values.astype(np.uint8)
class_names = class_df['name'].tolist()

# Setup SQLite DB
conn = sqlite3.connect('db.sqlite3', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, filename TEXT, date TEXT, summary TEXT)''')
conn.commit()

# Decode prediction mask to RGB
def decode_mask(mask):
    return colormap[mask]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Preprocess image
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(image, (256, 256))
            input_tensor = np.expand_dims(resized / 255.0, axis=0)

            # Predict
            pred = model.predict(input_tensor)[0]
            pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)
            pred_rgb = decode_mask(pred_mask)

            # Save predicted mask
            pred_path = os.path.join(UPLOAD_FOLDER, 'pred_' + filename)
            cv2.imwrite(pred_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

            # Summary
            unique, counts = np.unique(pred_mask, return_counts=True)
            total = pred_mask.size
            summary = ', '.join([f"{class_names[i]}: {counts[i]/total*100:.2f}%" for i in range(len(unique))])

            # Log to DB
            c.execute("INSERT INTO predictions (filename, date, summary) VALUES (?, ?, ?)",
                      (filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), summary))
            conn.commit()

            # return render_template('index.html', image_url=filepath, pred_url=pred_path, summary=summary)
            return render_template('index.html', image_url=filename, pred_url='pred_' + filename, summary=summary)


    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
