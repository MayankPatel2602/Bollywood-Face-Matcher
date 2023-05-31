import pickle

import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity

detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path,model,detector):
    img=cv2.imread(img_path)
    result = detector.detect_faces(img)
    x, y, width, height = result[0]['box']
    face = img[y:y + height, x:x + width]
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)

    face_array = face_array.astype('float32')
    face_array = face_array.astype('float32')

    expanded_image = np.expand_dims(face_array, axis=0)
    preprocessedd_image = preprocess_input(expanded_image)

    result = model.predict(preprocessedd_image).flatten()
    return result

def recommender(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('Which Bollywood Celebrity does your Face match?')
uploaded_img = st.file_uploader('Choose an image')

if uploaded_img is not None:
    # save img in directory
    if save_uploaded_image(uploaded_img):
        # load the image
        display_img = Image.open(uploaded_img)
        #extract features
        features = extract_features(os.path.join('uploads',uploaded_img.name),model,detector)
        #recommend
        index_pos=recommender(feature_list,features)
        st.text(index_pos)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        #display
        col1,col2 = st.columns(2)

        with col1:
            st.header('Your Uplaoded Image')
            st.image(display_img)

        with col2:
            st.header("Looks like " + predicted_actor)
            st.image(filenames[index_pos],width=300)


