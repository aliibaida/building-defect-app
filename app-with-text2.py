try:
    import streamlit as st
    from PIL import Image
    import random
    import cv2
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
except ModuleNotFoundError as e:
    print("Required module not found. Please ensure Streamlit, Pillow, OpenCV, and TensorFlow are installed.")
    raise e

# Load pre-trained model (example using a MobileNetV2 model fine-tuned for defect detection)
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = tf.keras.models.load_model("pretrained_defect_model.h5")  # Replace with actual model path
        return model
    except Exception as e:
        st.error("Error loading the model: " + str(e))
        return None

model = load_model()

# Function to classify defect severity using the pre-trained model
def classify_defect(image, explanation):
    try:
        if model is None:
            return "Error", None, "Model not loaded."

        # Preprocess the image
        image = image.resize((224, 224))  # Resize to model input size
        image_array = img_to_array(image)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        # Predict using the model
        prediction = model.predict(image_array)[0]
        defect_types = ["Crack", "Water Damage", "Corrosion", "No Defect"]
        detected_defect = defect_types[np.argmax(prediction)]
        severity_levels = ["Low", "Medium", "High"]

        if detected_defect == "No Defect":
            return detected_defect, None, "No action required."

        # Simple severity prediction based on confidence level (placeholder logic)
        confidence = prediction[np.argmax(prediction)]
        if confidence > 0.8:
            severity = "High"
        elif confidence > 0.5:
            severity = "Medium"
        else:
            severity = "Low"

        recommendation = {
            "Crack": "Monitor the crack for expansion and consult a structural engineer if it worsens.",
            "Water Damage": "Inspect for leaks and repair the source of water ingress.",
            "Corrosion": "Clean the affected area and apply anti-corrosion treatment."
        }.get(detected_defect, "No specific recommendation.")

        return detected_defect, severity, recommendation
    except Exception as e:
        st.error("Error during defect classification: " + str(e))
        return "Error", None, "Unable to classify the defect."

# Streamlit UI
try:
    st.title("Building Defect Detection App")
    st.write("Upload an image of a building element, and the app will detect any defects, categorize severity, and provide recommendations.")

    # Option to upload image
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    explanation = st.text_area("Provide a brief explanation of the photo (optional):")

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Processing...")
            defect, severity, recommendation = classify_defect(image, explanation)
            if defect == "No Defect":
                st.success("No defects detected.")
            else:
                st.error(f"Defect Detected: {defect}")
                st.warning(f"Severity: {severity}")
                st.info(f"Recommendation: {recommendation}")
        except Exception as e:
            st.error("Failed to process the uploaded image. Please upload a valid image.")

    # Admin Section Placeholder
    st.sidebar.title("Admin Section")
    st.sidebar.write("(Coming Soon) Upload and manage custom AI models.")

except Exception as e:
    print("An error occurred while running the Streamlit app:", e)
    raise e
