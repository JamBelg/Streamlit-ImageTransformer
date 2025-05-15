import streamlit as st
import cv2
import numpy as np

STATIC_DEFAULTS = {
    "color_mode": "Color (BGR)",
    "zoom": 1.0,
    "rotation_angle": 0,
    "brightness": 0,
    "contrast": 1.0,
    "blur": 0,
    "apply_canny": False
}

st.sidebar.write("This application uses OpenCV to transform a picture.")
uploaded_file = st.sidebar.file_uploader("Pick a file to upload a picture", type=["jpg", "jpeg", "png"])

st.title("Image Transformer App")

st.markdown(
    """
    <div style="text-align: left; font-size: 0.95em; color: c; margin-top: -10px; margin-bottom: 30px;">
        <a href="https://www.jamelbelgacem.com" target="_blank" style="text-decoration: none; color: blue;">
        Jamel Belgacem</a> © 2025 
    </div>
    <hr style="margin-bottom: 25px;">
    """,
    unsafe_allow_html=True
)

st.write("Upload an image and adjust transformation settings from the sidebar.")


# --- Reset function ---
def reset_all():
    for key, val in STATIC_DEFAULTS.items():
        st.session_state[key] = val
    if 'original_shape' in st.session_state:
        st.session_state['width'] = st.session_state['original_shape'][1]
        st.session_state['height'] = st.session_state['original_shape'][0]

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_bytes_np = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes_np, cv2.IMREAD_UNCHANGED)

    orig_h, orig_w = image.shape[:2]
    st.session_state['original_shape'] = (orig_h, orig_w)

    # Initialize session state variables
    if 'width' not in st.session_state:
        st.session_state['width'] = orig_w
    if 'height' not in st.session_state:
        st.session_state['height'] = orig_h
    if 'color_mode' not in st.session_state:
        st.session_state['color_mode'] = STATIC_DEFAULTS['color_mode']
    if 'zoom' not in st.session_state:
        st.session_state['zoom'] = STATIC_DEFAULTS['zoom']
    if 'rotation_angle' not in st.session_state:
        st.session_state['rotation_angle'] = STATIC_DEFAULTS['rotation_angle']
    if 'brightness' not in st.session_state:
        st.session_state['brightness'] = STATIC_DEFAULTS['brightness']
    if 'contrast' not in st.session_state:
        st.session_state['contrast'] = STATIC_DEFAULTS['contrast']
    if 'blur' not in st.session_state:
        st.session_state['blur'] = STATIC_DEFAULTS['blur']
    if 'apply_canny' not in st.session_state:
        st.session_state['apply_canny'] = STATIC_DEFAULTS['apply_canny']

    # --- Widgets (must come after session state init/reset) ---
    color_mode = st.sidebar.selectbox(
        "Color mode", ["Color (BGR)", "Grayscale", "Bluescale", "Greenscale", "Redscale"],
        key="color_mode"
    )

    width = st.sidebar.slider("Resize Width",
                              min_value=50,
                              max_value=3840,
                              step=1,
                              key="width")
    height = st.sidebar.slider("Resize Height",
                               min_value=50,
                               max_value=2160,
                               step=1,
                               key="height")
    zoom = st.sidebar.slider("Zoom",
                             min_value=0.1,
                             max_value=3.0,
                             step=0.1,
                             key="zoom")
    rotation_angle = st.sidebar.slider("Rotation (degrees)",
                                       min_value=-180,
                                       max_value=180,
                                       step=1,
                                       key="rotation_angle")
    brightness = st.sidebar.slider("Brightness",
                                   min_value=-100,
                                   max_value=100,
                                   step=1,
                                   key="brightness")
    contrast = st.sidebar.slider("Contrast",
                                 min_value=0.5,
                                 max_value=3.0,
                                 step=0.1,
                                 key="contrast")
    blur = st.sidebar.slider("Blur intensity",
                             min_value=0,
                             max_value=20,
                             step=1,
                             key="blur")
    apply_canny = st.sidebar.checkbox("Apply Canny Edge Detection", key="apply_canny")

    # --- Reset button ---
    st.sidebar.button("🔄 Reset All", on_click=reset_all)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, caption="Original Image", use_container_width=True, channels="BGR")

    with col2:
        st.subheader("After transformations")
        transformed = image.copy()

        # Apply color mode
        if color_mode == "Grayscale":
            transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
        elif color_mode == "Bluescale":
            transformed[:, :, 1] = 0
            transformed[:, :, 2] = 0
        elif color_mode == "Greenscale":
            transformed[:, :, 0] = 0
            transformed[:, :, 2] = 0
        elif color_mode == "Redscale":
            transformed[:, :, 0] = 0
            transformed[:, :, 1] = 0

        # Resize
        transformed = cv2.resize(transformed, (width, height))

        # Zoom
        cx, cy = transformed.shape[1] // 2, transformed.shape[0] // 2
        zoom_w, zoom_h = int(cx / zoom), int(cy / zoom)
        zoomed = transformed[
            max(0, cy - zoom_h):min(transformed.shape[0], cy + zoom_h),
            max(0, cx - zoom_w):min(transformed.shape[1], cx + zoom_w)
        ]
        transformed = cv2.resize(zoomed, (width, height))

        # Brightness & Contrast
        transformed = cv2.convertScaleAbs(transformed, alpha=contrast, beta=brightness)

        # Rotation
        center = (width // 2, height // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        transformed = cv2.warpAffine(transformed, rot_matrix, (width, height))

        # Blur
        if blur > 0:
            ksize = blur if blur % 2 == 1 else blur + 1
            transformed = cv2.GaussianBlur(transformed, (ksize, ksize), 0)

        # Canny Edge Detection
        if apply_canny:
            if len(transformed.shape) == 3:
                transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
            transformed = cv2.Canny(transformed, 100, 200)

        # Display result
        st.image(
            transformed,
            caption="Transformed Image",
            use_container_width=True,
            channels="GRAY" if len(transformed.shape) == 2 else "BGR"
        )
        # Save button and download functionality
        is_success, buffer = cv2.imencode(".jpg", transformed)
        if is_success:
            st.download_button(
                label="💾 Download Transformed Image",
                data=buffer.tobytes(),
                file_name="transformed_image.jpg",
                mime="image/jpeg"
            )
else:
    st.warning("Please upload an image to see the transformations.")


