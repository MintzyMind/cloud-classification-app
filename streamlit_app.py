import streamlit as st
import torch
from PIL import Image
from prediction import pred_class
import numpy as np

# Set title
st.title('Cloud Classification')

# Set Header
st.header('Please upload a picture')

# Load Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Use torch.load to load the model if it's a saved PyTorch model
model_path = 'mobilenetv3_large_100_checkpoint_fold0.pt'
try:
    model = torch.load(model_path, map_location=device)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Display image & Prediction
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    class_name = ['Altocumulus', 'Altostratus', 'Cirrocumulus', 'Cirrostratus', 'Cirrus', 'Contrails', 'Cumulonimbus', 'Cumulus', 'Nimbostratus', 'Stratocumulus', 'Stratus']

    if st.button('Prediction'):

        if 'model' not in locals():
            st.error("Model not loaded successfully. Please check the model file.")
        else:
            # Prediction class
            probli = pred_class(model, image, class_name)

            st.write("## Prediction Result")
            # Get the index of the maximum value in probli[0]
            max_index = np.argmax(probli[0])

            # Iterate over the class_name and probli lists
            for i in range(len(class_name)):
                # Set the color to blue if it's the maximum value, otherwise use the default color
                color = "blue" if i == max_index else "black"
                st.write(
                    f"## <span style='color:{color}'>{class_name[i]} : {probli[0][i] * 100:.2f}%</span>", unsafe_allow_html=True)
