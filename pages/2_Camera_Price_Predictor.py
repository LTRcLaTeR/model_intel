import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from pathlib import Path

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent.parent / "Sony_Camera_Predict.joblib"
    return joblib.load(model_path)

@st.cache_resource
def load_encoder():
    encoder_path = Path(__file__).resolve().parent.parent / "model_encoder.joblib"
    return joblib.load(encoder_path)

@st.cache_data
def load_ref():
    ref_path = Path(__file__).resolve().parent.parent / "train_Machine" / "Sony_Price_REF.csv"
    return pd.read_csv(ref_path)

model = load_model()
le = load_encoder()
ref_df = load_ref()

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

    st.markdown("##### 📏 Ridge Regression")
    st.markdown("""
    เป็นการปรับปรุงจาก Linear Regression โดยเพิ่มตัวแปรควบคุม (L2 Regularization) เพื่อป้องกันค่าน้ําหนัก
ตัวแปรไม่ให้โดดเกินไป ช่วยให้โมเดลมีฐานการทํานายที่เสถียรและยึดโยงกับราคาอ้างอิงตลาด (REF) ได้อย่างมั่นคง
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
    ไปที่แท็บ **🚀 ใช้งานโมเดล** แล้วเลือกรุ่นกล้องจากรายการ จากนั้นระบุข้อมูลเพิ่มเติม 2 รายการ ดังนี้:
    """)
    st.markdown("""
    | ช่องกรอก | คำอธิบาย |
    |---|---|
    | **เลือกรุ่นกล้อง** | เลือกรุ่นจากรายการ Sony_Price_REF — ราคาอ้างอิงตลาดจะถูกโหลดอัตโนมัติ |
    | **อายุ / Shutter / สภาพ** | ระบุข้อมูลสภาพจริงของกล้อง |
    | **มีกล่อง** | ระบุว่ามีกล่องและอุปกรณ์ครบหรือไม่ |
    """)
    st.markdown("""
    เมื่อเลือกรุ่นแล้ว กดปุ่ม **🔮 ทำนายราคา** โมเดลจะแสดง **ราคาประเมินที่เหมาะสม** พร้อมเปรียบเทียบกับราคาอ้างอิงทันที
    """)
    st.info("💡 ราคาที่ได้เป็นการประมาณการจากข้อมูลตลาด ณ วันที่ 15 มีนาคม 2569 ควรใช้เป็นข้อมูลอ้างอิงเท่านั้น")

with tab_model:
    camera_names = ref_df["model"].tolist()
    selected = st.selectbox("เลือกรุ่นกล้อง Sony", camera_names)

    row = ref_df[ref_df["model"] == selected].iloc[0]

    max_age = int(row["age_year"])
    default_shutter = int(row["shutter_claim"])
    default_condition = row["condition"]

    condition_options = ["excellent", "very_good", "good", "fair", "poor"]
    default_cond_idx = condition_options.index(default_condition) if default_condition in condition_options else 2

    col1, col2, col3 = st.columns(3)
    with col1:
        age_year = st.number_input("อายุ (ปี)", min_value=0, max_value=max_age, value=max_age,
                                   help=f"กล้องรุ่นนี้เปิดตัวมา {max_age} ปี (สูงสุด)")
    with col2:
        shutter_count = st.number_input("Shutter Count", min_value=0, max_value=9999999, value=default_shutter)
    with col3:
        condition = st.selectbox("สภาพ", condition_options, index=default_cond_idx,
                                 format_func=lambda x: {"excellent": "Excellent ✨", "very_good": "Very Good 👍",
                                                         "good": "Good 👌", "fair": "Fair 🤏", "poor": "Poor 😵"}.get(x, x))

    st.divider()

    has_box = st.selectbox("มีกล่อง / อุปกรณ์ครบ?", [0, 1], format_func=lambda x: "มี ✅" if x == 1 else "ไม่มี ❌")

    cond_map = {"excellent": 10, "very_good": 8, "good": 6, "fair": 4, "poor": 2}
    cond_score    = cond_map.get(condition, 5)
    age_score     = 10 if age_year < 3 else 5
    shutter_score = 10 if shutter_count < 20000 else 5
    box_score     = 10 if has_box == 1 else 0

    ref_p = float(re.sub(r'[^0-9.]', '', str(row['price_used'])))

    with st.expander("ดูคะแนน Features ที่คำนวณได้"):
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("cond_score", cond_score)
        sc2.metric("age_score", age_score)
        sc3.metric("shutter_score", shutter_score)
        sc4.metric("box_score", box_score)
        st.caption(f"ราคาอ้างอิงตลาด (REF): {ref_p:,.0f} บาท")

    if st.button("🔮 ทำนายราคา"):
        m_clean = selected.replace('Sony ', '')
        try:
            m_idx = le.transform([m_clean])[0]
        except Exception:
            m_idx = 0

        input_df = pd.DataFrame(
            [[m_idx, ref_p, age_year, shutter_count, has_box, cond_score, age_score, shutter_score, box_score]],
            columns=['model_encoded', 'price_used_ref_numeric', 'age_year', 'shutter_count',
                     'has_box', 'cond_score', 'age_score', 'shutter_score', 'box_score']
        )
        pred = model.predict(input_df)[0]
        final_price = (pred * 0.3) + (ref_p * 0.7)

        st.success(f"ราคาประเมิน: **{final_price:,.0f} บาท**")
