import streamlit as st
import joblib
import numpy as np
from pathlib import Path

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent.parent / "Sony_Camera_Predict.joblib"
    return joblib.load(model_path)

model = load_model()

st.title("📷 Camera Price Predictor")

tab_article, tab_model = st.tabs(["📝 บทความ", "🚀 ใช้งานโมเดล"])

with tab_article:
    st.header("แนวทางการพัฒนาโมเดลคาดการณ์ราคากล้อง Sony")

    st.subheader("📦 การเตรียมข้อมูล")
    st.markdown("""
    โปรเจกต์นี้เตรียมข้อมูลโดยให้ ChatGPT สร้างไฟล์ `.csv` ขึ้นมา โดยกำหนดเงื่อนไขข้อมูลต่าง ๆ อ้างอิงจากราคาตลาดกล้องมือสองจากกลุ่มซื้อขายในแพลตฟอร์ม **Facebook** และ **Facebook Marketplace** รวมถึงประสบการณ์การซื้อ/ขายกล้องมือสองของตนเอง

    เมื่อได้ Dataset แล้ว นำมาตรวจสอบความถูกต้องโดยใช้ข้อมูลจริงเพื่อคัดกรองอีกขั้นตอน หลังจากนั้นนำข้อมูลที่ได้มาทำให้เกิด Missing เพื่อเตรียมข้อมูลให้พร้อมสำหรับการ **Data Cleansing** และเตรียมข้อมูลอ้างอิงของจริงโดยประมาณ จากราคาตลาดของวันที่ **15 มีนาคม 2569** ทำเป็นไฟล์ `Sony_Price_REF.csv` เพื่อไม่ให้โมเดลเกิดความคลาดเคลื่อน
    """)

    st.subheader("🧠 ทฤษฎีของอัลกอริทึม")
    st.markdown("""
    โปรเจกต์นี้เลือกใช้เทคนิค **VotingRegressor** เพื่อรวมความสามารถของโมเดลที่แตกต่างกัน 4 ประเภท ได้แก่:
    """)

    st.markdown("##### 🌲 Random Forest")
    st.markdown("""
    ใช้ทฤษฎี **Bagging** สร้าง Decision Trees หลายต้นแบบขนานกัน และนำค่าเฉลี่ยมาตอบ ช่วยลดความแปรปรวน (Variance) และป้องกันการที่โมเดลจำข้อมูลแม่นเกินไป (Overfitting)
    """)

    st.markdown("##### 📈 Gradient Boosting")
    st.markdown("""
    ใช้ทฤษฎี **Boosting** สร้างต้นไม้ทีละต้น โดยต้นใหม่จะพยายามแก้ไขความผิดพลาด (Residual Error) ที่เกิดขึ้นของต้นก่อนหน้า ทำให้มีความแม่นยำ (Accuracy) สูงมากในรูปแบบของข้อมูลที่ซับซ้อน
    """)

    st.markdown("##### 📏 Linear Regression")
    st.markdown("""
    ใช้ทฤษฎี **Statistical Modeling** หาความสัมพันธ์เชิงเส้นระหว่างตัวแปรอิสระ (เช่น อายุเครื่อง) กับราคา ช่วยให้โมเดลมีฐานการทำนายที่เสถียรตามแนวโน้มตลาดพื้นฐาน
    """)

    st.markdown("##### 🌳 Extra Trees")
    st.markdown("""
    คล้ายกับ Random Forest แต่จะใช้การ **Random Split** จุดตัดสินใจอย่างอิสระ ช่วยลดสัญญาณรบกวน (Noise) ในข้อมูลได้ดีกว่า และเพิ่มความหลากหลายให้กลุ่มโมเดล
    """)

    st.subheader("⚙️ ขั้นตอนการพัฒนาโมเดล (Development Process)")

    st.markdown("##### 1. Data Cleaning & Integration")
    st.markdown("""
    นำข้อมูลจาก Dataset Sony Camera มาเชื่อมโยงกับ Price Reference โดยใช้ชื่อรุ่นเป็นกุญแจหลัก จัดการค่าที่ผิดปกติ เช่น การลบอักขระภาษาไทยออกจากราคา และการเติมค่าว่าง (Imputation) ในส่วนของ Shutter Count
    """)

    st.markdown("##### 2. Feature Engineering (Scoring Logic)")
    st.markdown("""
    แปลงข้อมูลคุณภาพ (Qualitative) ให้เป็นเชิงปริมาณ (Quantitative) เช่น การแปลงสภาพกล้อง (Condition) เป็น `cond_score` (2–10 คะแนน) คำนวณคะแนนเสริม เช่น `age_score` และ `shutter_score` เพื่อให้โมเดลเข้าใจ "สุขภาพ" ของอุปกรณ์ได้ง่ายขึ้น
    """)

    st.markdown("##### 3. Data Splitting")
    st.markdown("""
    แบ่งข้อมูลเป็น **Training Set (80%)** สำหรับสอนโมเดล และ **Test Set (20%)** สำหรับวัดผลการทำนายว่าแม่นยำจริงหรือไม่
    """)

    st.markdown("##### 4. Model Training & Ensemble")
    st.markdown("""
    ฝึกสอนโมเดลทั้ง 4 ประเภทพร้อมกัน และใช้ **VotingRegressor** ทำหน้าที่เป็น "กรรมการ" นำผลลัพธ์จากทุกโมเดลมาหาค่าเฉลี่ย
    """)

    st.markdown("##### 5. Model Export")
    st.markdown("""
    บันทึกโครงสร้างของโมเดลที่ฝึกเสร็จแล้ว ลงในไฟล์ `Sony_Camera_Predict.joblib` เพื่อให้สามารถนำไปใช้งานได้ทันทีโดยไม่ต้องเทรนโมเดลใหม่
    """)

    st.subheader("📚 แหล่งอ้างอิงข้อมูลที่นำมาใช้")
    st.markdown("""
    - [Facebook Marketplace](https://www.facebook.com/marketplace/)
    - [กลุ่ม ซื้อขายกล้องมือสอง](https://www.facebook.com/groups/113550375846414)
    - [กลุ่ม ซื้อขายกล้องมือสอง (Camera 2 hand Market)](https://www.facebook.com/groups/366990763511834)
    - [กลุ่ม Camera Market](https://www.facebook.com/groups/483481148331248)
    - [กลุ่ม กล้องมือสอง](https://www.facebook.com/groups/305295316191198)
    - [กลุ่ม Camera All Market (ซื้อขายกล้องมือสอง)](https://www.facebook.com/groups/483481148331248)
    - [กลุ่ม Sony Market Thailand (ขายกล้องเลนส์)](https://www.facebook.com/groups/483481148331248)
    - [กลุ่ม ห้องซื้อขายกล้อง เลนส์อุปกรณ์ Sony](https://www.facebook.com/groups/385602655192591)
    - [กลุ่ม Sony Camera Market Thailand](https://www.facebook.com/groups/taradsony)
    """)

    st.divider()
    st.subheader("🚀 วิธีการใช้งานโมเดล")
    st.markdown("""
    ไปที่แท็บ **🚀 ใช้งานโมเดล** แล้วกรอกข้อมูลกล้องที่ต้องการประเมินราคา ดังนี้:
    """)
    st.markdown("""
    | ช่องกรอก | คำอธิบาย |
    |---|---|
    | **Age (years)** | อายุกล้องเป็นปี นับจากปีที่ผลิต |
    | **Shutter Count** | จำนวนครั้งที่กดชัตเตอร์สะสม |
    | **Has Box** | มีกล่องหรืออุปกรณ์ครบชุดหรือไม่ (0 = ไม่มี, 1 = มี) |
    | **Condition Score** | คะแนนสภาพภายนอก 1–10 (10 = ใหม่มาก) |
    | **Age Score** | คะแนนความเก่าของเครื่อง 1–10 (10 = อายุน้อย) |
    | **Shutter Score** | คะแนนการใช้งานชัตเตอร์ 1–10 (10 = ใช้น้อยมาก) |
    | **Price Score** | คะแนนภาพรวมความคุ้มค่าราคา 1–10 |
    | **Box Score** | คะแนนความสมบูรณ์ของกล่องและอุปกรณ์เสริม 1–10 |
    """)
    st.markdown("""
    เมื่อกรอกข้อมูลครบแล้ว กดปุ่ม **Predict** โมเดลจะแสดง **ราคาประเมินที่เหมาะสม** สำหรับกล้อง Sony มือสองทันที
    """)
    st.info("💡 ราคาที่ได้เป็นการประมาณการจากข้อมูลตลาด ณ วันที่ 15 มีนาคม 2569 ควรใช้เป็นข้อมูลอ้างอิงเท่านั้น")

with tab_model:
    age_year = st.number_input("Age (years)", 0, 30, 3)
    shutter_count = st.number_input("Shutter Count", 0, 200000, 10000)
    has_box = st.selectbox("Has Box", [0, 1])
    cond_score = st.slider("Condition Score", 1, 10, 8)
    age_score = st.slider("Age Score", 1, 10, 7)
    shutter_score = st.slider("Shutter Score", 1, 10, 7)
    price_score = st.slider("Price Score", 1, 10, 5)
    box_score = st.slider("Box Score", 1, 10, 5)

    if st.button("Predict"):
        data = np.array([[age_year, shutter_count, has_box, cond_score, age_score, shutter_score, price_score, box_score]])
        price = model.predict(data)

        st.success(f"Predicted Price: {price[0]:,.0f} บาท")
