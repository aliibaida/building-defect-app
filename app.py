try:
    import streamlit as st
    from PIL import Image
    import random
except ModuleNotFoundError as e:
    print("Required module not found. Please ensure Streamlit and Pillow are installed.")
    raise e

# Function to classify defect severity
def classify_defect(image):
    # Mock AI logic for demonstration purposes
    defect_types = ["Crack", "Water Damage", "Corrosion", "No Defect"]
    severity_levels = ["Low", "Medium", "High"]

    detected_defect = random.choice(defect_types)
    if detected_defect == "No Defect":
        return detected_defect, None, "No action required."
    
    severity = random.choice(severity_levels)
    recommendation = {
        "Crack": "Monitor the crack for expansion and consult a structural engineer if it worsens.",
        "Water Damage": "Inspect for leaks and repair the source of water ingress.",
        "Corrosion": "Clean the affected area and apply anti-corrosion treatment."
    }.get(detected_defect, "No specific recommendation.")

    return detected_defect, severity, recommendation

# Streamlit UI
try:
    st.title("Building Defect Detection App")
    st.write("Upload an image of a building element, and the app will detect any defects, categorize severity, and provide recommendations.")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        st.write("Processing...")
        defect, severity, recommendation = classify_defect(image)
        
        if defect == "No Defect":
            st.success("No defects detected.")
        else:
            st.error(f"Defect Detected: {defect}")
            st.warning(f"Severity: {severity}")
            st.info(f"Recommendation: {recommendation}")

    # Admin Section Placeholder
    st.sidebar.title("Admin Section")
    st.sidebar.write("(Coming Soon) Upload and manage custom AI models.")

except Exception as e:
    print("An error occurred while running the Streamlit app:", e)
    raise e
