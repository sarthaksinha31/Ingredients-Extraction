import cv2
import pytesseract 
import re

from similarity import get_cosine

from sentence_transformers import SentenceTransformer, util

from io import BytesIO, StringIO
from typing import Union
from PIL import Image
import pandas as pd
import streamlit as st
import numpy as np

# embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')

STYLE = """
<style>
img {
    max-width: 80%;
    position: relative;
    transform: rotate(90deg);
    left: 700px;

}
</style>
"""



def ingredients_extraction(img_content):
    embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
    # img = cv2.imread(img_content)
    img  = img_content

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    im2 = img.copy()
    i = 0
    text_list = []
    for cnt in contours[0:len(contours)-1]: 
        x, y, w, h = cv2.boundingRect(cnt) 
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 

        cropped = im2[y:y + h, x:x + w] 
        text = pytesseract.image_to_string(cropped) 

        if re.findall(r"\w+",text) != []:
            text_split = text.split(",")
            if text_split != [] and len(text_split) > 1:

                text_list.append(text)

    query_embedding = embedder.encode(text_list)

    scores = list(map(get_cosine,query_embedding))
    # print(scores)
    try:
        max_index = scores.index(max(scores))
        return text_list[max_index]

    except:
        text = pytesseract.image_to_string(im2)
        # print(text)
        text_list_ = re.split("\n{2,}",text)
        # print(text_list_)
        query_embedding = embedder.encode(text_list_)
        scores = list(map(get_cosine,query_embedding))
        max_index = scores.index(max(scores))
        return text_list_[max_index]


def main():
    
    # st.info(__doc__)
    st.info("GET THE INGREDIENTS")
    st.markdown(STYLE, unsafe_allow_html=True)
 
    uploaded_file = st.file_uploader("Upload an image of a food packet", type="jpg")
    show_file = st.empty()
    if isinstance(uploaded_file, BytesIO):
        show_file.image(uploaded_file)
    if uploaded_file is not None:
    # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        ingredient = ingredients_extraction(opencv_image)

        st.info(ingredient)
        uploaded_file.close()

st.set_option('deprecation.showfileUploaderEncoding', False)
main()