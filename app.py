import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
import matplotlib.pyplot as plt
st.title("Garbage classification")
model = tf.keras.models.load_model("classifyWaste.h5")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

# map_dict = {0: 'plastic',
#             1: 'e-waste'}

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
          
          test_image = image.load_img(uploaded_file, target_size = (224,224))
        #   plt.axis("off")
        #   plt.imshow(test_image)
        #   plt.show()
        
          test_image = image.img_to_array(test_image) / 255
          test_image = np.expand_dims(test_image, axis=0)
          predicted_array = model.predict(test_image)
        #   print(predicted_array)

          class_one = predicted_array > 0.5

        #   print(class_one[0][0])
        #   print(type(class_one[0][0]))

          if(class_one[0][0]==True):
                st.title("The waste material is plastic")
          
          else:
                st.title("The waste material is e-waste")
          
        #   prediction = model.predict(img_reshape).argmax()
        #   st.title("Predicted Label for the image is {}".format(map_dict [prediction]))