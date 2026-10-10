
import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from PIL import Image

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="Medical Image AI",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical Image Analysis AI")
st.write(
    "Upload a medical image to receive an "
    "AI-generated explanation for educational purposes."
)

if not API_KEY:
    st.error("Gemini API key not found. Check your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

uploaded_file = st.file_uploader(
    "Upload a medical image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width="stretch")

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            try:
                prompt = """
                Describe the uploaded medical image cautiously.

                If it is a recognizable medical image, provide:
                1. The type of image, if identifiable.
                2. Visible observations.
                3. Possible interpretations and uncertainty.
                4. Image quality or limitations.
                5. Questions to discuss with a qualified clinician.

                Do not claim a confirmed diagnosis, rule out disease,
                or recommend treatment. State that this output is
                AI-generated and requires professional interpretation.
                If the image is not recognizable, say so.
                """

                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[prompt, image]
                )

                st.subheader("AI-Generated Report")
                st.write(response.text)

            except Exception as error:
                st.error(f"Analysis failed: {error}")

st.warning(
    "Educational prototype only. This is not a medical "
    "diagnostic tool. Do not upload identifiable patient data."
)
