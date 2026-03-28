import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Patch Dense.from_config to ignore quantization_config
_orig_dense_from_config = tf.keras.layers.Dense.from_config

@classmethod
def _patched_dense_from_config(cls, config):
    config.pop("quantization_config", None)
    return _orig_dense_from_config(config)

tf.keras.layers.Dense.from_config = _patched_dense_from_config

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent.parent / "my_model.keras"
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

classes = ["blue","green","purple","red"]

st.title("🎨 Card Color AI")

tab_article, tab_model = st.tabs(["📝 บทความ", "🚀 ใช้งานโมเดล"])

with tab_article:
    st.header("แนวทางการพัฒนาโมเดลจำแนกสีการ์ด (Card Color AI)")

    st.subheader("📦 การเตรียมข้อมูล")
    st.markdown("""
    โปรเจกต์นี้เตรียมข้อมูลโดยการแคปรูปภาพการ์ดจากเว็บไซต์ [bangbon.app](https://bangbon.app/) ซึ่งรูปภาพที่ได้มามีขนาดไม่เท่ากัน จึงต้องมีขั้นตอนการเตรียมข้อมูลเพิ่มเติม ดังนี้:

    - **Resize** — ปรับขนาดรูปภาพทั้งหมดให้เป็นขนาดเดียวกันที่ **128×128 พิกเซล** เพื่อให้โมเดลสามารถรับ Input ที่มี Shape คงที่ได้
    - **Normalization** — แปลงค่าพิกเซลจาก 0–255 ให้อยู่ในช่วง 0–1 เพื่อช่วยให้โมเดลเรียนรู้ได้เร็วและเสถียรขึ้น
    - **Labeling** — จัดแบ่งรูปภาพตามสีของการ์ดออกเป็น 4 คลาส ได้แก่ 🔴 Red, 🔵 Blue, 🟢 Green, 🟣 Purple
    """)

    st.subheader("🧠 ทฤษฎีของอัลกอริทึม")
    st.markdown("""
    โปรเจกต์นี้เลือกใช้ **Convolutional Neural Network (CNN)** ซึ่งเป็นโมเดล Deep Learning ที่ออกแบบมาเพื่อประมวลผลข้อมูลภาพโดยเฉพาะ โดยมีองค์ประกอบหลัก ดังนี้:
    """)

    st.markdown("##### 🔍 Convolutional Layer")
    st.markdown("""
    ทำหน้าที่สกัดคุณลักษณะ (Feature Extraction) จากรูปภาพ โดยใช้ **Filter/Kernel** เลื่อนไปบนภาพเพื่อตรวจจับรูปแบบต่าง ๆ เช่น ขอบ สี พื้นผิว ทำให้โมเดลเข้าใจลักษณะเฉพาะของแต่ละสีการ์ดได้
    """)

    st.markdown("##### 📐 Pooling Layer")
    st.markdown("""
    ลดขนาดของ Feature Map ที่ได้จาก Convolutional Layer เพื่อลดจำนวน Parameter และป้องกัน Overfitting โดยยังคงรักษาคุณลักษณะสำคัญของภาพไว้
    """)

    st.markdown("##### 🧩 Fully Connected Layer (Dense)")
    st.markdown("""
    นำ Feature ที่สกัดได้มารวมกัน แล้วส่งผ่าน **Softmax Activation** เพื่อจำแนกภาพออกเป็น 4 คลาสสี พร้อมค่าความน่าจะเป็น (Probability) ของแต่ละคลาส
    """)

    st.subheader("⚙️ ขั้นตอนการพัฒนาโมเดล (Development Process)")

    st.markdown("##### 1. Data Collection")
    st.markdown("""
    แคปรูปภาพการ์ดแต่ละสีจากเว็บไซต์ [bangbon.app](https://bangbon.app/) แล้วจัดเก็บแยกตามโฟลเดอร์ของแต่ละสี
    """)

    st.markdown("##### 2. Data Preprocessing")
    st.markdown("""
    ปรับขนาดรูปภาพให้เท่ากันที่ 128×128 พิกเซล, Normalize ค่าพิกเซลให้อยู่ในช่วง 0–1 และแปลง Label ให้อยู่ในรูปแบบที่โมเดลเข้าใจ
    """)

    st.markdown("##### 3. Data Splitting")
    st.markdown("""
    แบ่งข้อมูลเป็น **Training Set** สำหรับสอนโมเดล และ **Validation/Test Set** สำหรับวัดผลการจำแนกว่าแม่นยำจริงหรือไม่
    """)

    st.markdown("##### 4. Model Training")
    st.markdown("""
    ฝึกสอนโมเดล CNN ด้วยข้อมูลภาพการ์ด โดยโมเดลจะเรียนรู้คุณลักษณะของแต่ละสีผ่านการปรับ Weight ในแต่ละ Epoch จนกว่าจะได้ความแม่นยำที่ต้องการ
    """)

    st.markdown("##### 5. Model Export")
    st.markdown("""
    บันทึกโมเดลที่ฝึกเสร็จแล้วลงในไฟล์ `my_model.keras` เพื่อให้สามารถนำไปใช้งานได้ทันทีโดยไม่ต้องเทรนโมเดลใหม่
    """)

    st.subheader("📚 แหล่งอ้างอิงข้อมูลที่นำมาใช้")
    st.markdown("""
    - [bangbon.app](https://bangbon.app/) — แหล่งข้อมูลรูปภาพการ์ดที่ใช้ในการเทรนโมเดล
    """)

with tab_model:
    st.markdown("🔗 ค้นหาการ์ดใบอื่นมาทำนายได้ที่ [bangbon.app](https://bangbon.app/)")
    st.divider()

    st.subheader("📤 อัปโหลดรูปการ์ด")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, width=250)

        img_resized = img.resize((128,128))
        img_array = np.array(img_resized)/255
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0]
        st.success(f"🎯 This card is: **{classes[np.argmax(pred)]}**")

    st.divider()
    st.subheader("🃏 ตัวอย่างการ์ดแต่ละสี")
    st.caption("กดปุ่มด้านล่างเพื่อทดลองทำนายสีการ์ดตัวอย่าง")

    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    example_colors = {"🔴 Red": "red.png", "🔵 Blue": "blue.png", "🟢 Green": "green.png", "🟣 Purple": "purple.png"}

    cols = st.columns(4)
    for i, (label, filename) in enumerate(example_colors.items()):
        img_path = examples_dir / filename
        with cols[i]:
            if img_path.exists():
                st.image(str(img_path), caption=label, width=128)
                if st.button(f"ทำนาย {label}", key=f"ex_{filename}"):
                    ex_img = Image.open(img_path).convert("RGB").resize((128,128))
                    ex_array = np.expand_dims(np.array(ex_img)/255, axis=0)
                    ex_pred = model.predict(ex_array)[0]
                    st.success(f"🎯 **{classes[np.argmax(ex_pred)]}**")
