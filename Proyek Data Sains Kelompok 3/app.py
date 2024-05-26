import streamlit as st
from PIL import Image

def main():
    st.title("Bank Customer Churn Prediction and Visualization")
    image = Image.open('/content/churn bank prediction.png')
    st.image(image)
    #st.sidebar.title("Navigation")
    #st.sidebar.markdown("[Prediction](Prediction)", unsafe_allow_html=True)
    #st.sidebar.markdown("[Visualization](Visualization)", unsafe_allow_html=True)
    st.write("**Pendahuluan**")
    latar_belakang = """
    Dalam era digital, industri perbankan menghadapi peningkatan churn, yaitu 
    kehilangan pelanggan ke bank lain, akibat persaingan ketat, perubahan preferensi 
    konsumen, dan teknologi yang memudahkan perbandingan layanan. Churn ini mengurangi 
    laba dan memerlukan prediksi tepat untuk retensi pelanggan. Pelanggan adalah aset penting, 
    sehingga menjaga loyalitas melalui Customer Relationship Management (CRM) menjadi krusial. 
    CRM menekankan pentingnya hubungan efektif dengan pelanggan untuk kesuksesan bisnis. 
    Fokus kami adalah memprediksi churn menggunakan model machine learning untuk mengurangi kehilangan 
    nasabah dan mempertahankan pelanggan berharga.
    """
    st.write(latar_belakang)
    st.write("**Dataset dan Machine Learning**")
    dataset = """
    Dataset yang kami gunakan berasal dari sumber data Kaggle. Kami memilih dataset 
    bernama “ Bank Churn Data Exploration And Churn Prediction ”.
    Dan untuk Machine Learning nya kami memakai Machine Learning
    bernama XGBoost.
    """
    st.write(dataset)

if __name__ == "__main__":
    main()
