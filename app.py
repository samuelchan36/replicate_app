import streamlit as st
import replicate
import requests
from io import BytesIO

# Streamlit UI setup
st.title("Image Generator with Replicate")

# User input for prompt
prompt = st.text_input("Enter your prompt:", "a beautiful castle illustration")

# Example prompt below the input box
st.markdown("**Example:**\n\nClaire as Spider-Man at the top of a building, looking forward to the camera.\n\n(Keywords: Claire or Max; Character: specify character or costume; Place: specify location; Camera: specify camera position)")

# Dropdown for LoRA selection
hf_lora = st.selectbox(
    "Choose a LoRA model:",
    ("Samuel6391/max-lora", "Samuel6391/clairechan-lora"),
    index=0
)

# Button to generate the image
if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a valid prompt.")
    else:
        with st.spinner("Generating image..."):
            try:
                # Define input parameters for the model
                input_params = {
                    "prompt": prompt,
                    "hf_lora": hf_lora
                }

                # Make a prediction
                output = replicate.run(
                    "lucataco/flux-dev-lora:091495765fa5ef2725a175a57b276ec30dc9d39c22d30410f2ede68a3eab66b3",
                    input=input_params
                )

                # Display the generated image
                if output:
                    image_url = output[0]
                    response = requests.get(image_url)
                    image = BytesIO(response.content)
                    st.image(image, caption="Generated Image", use_column_width=True)
                    # Download button
                    st.download_button(
                        label="Download Image",
                        data=image,
                        file_name="generated_image.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"An error occurred: {e}")
