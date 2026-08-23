import os
import requests
import pandas as pd
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecasting")
st.caption("Predict Product_Store_Sales_Total for a product-store combination.")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# data input from users
with tab1:
    st.subheader("Enter product & store details")

    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5, step=0.1)
        product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.3f")
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0, step=1.0)
        product_id_char = st.selectbox("Product Category Code", ["FD", "DR", "NC"], help="FD=Food, DR=Drinks, NC=Non-Consumable")

    with col2:
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Food Mart", "Supermarket Type1", "Supermarket Type2", "Departmental Store"])
        store_age = st.number_input("Store Age (years)", min_value=0, value=15, step=1)
        product_type_cat = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

    if st.button("Predict Sales", type="primary"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": product_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age,
            "Product_Type_Category": product_type_cat,
        }
        try:
            resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=15)
            if resp.status_code == 200:
                pred = resp.json()["predicted_sales"]
                st.success(f"Predicted Sales: **{pred:,.2f}**")
            else:
                st.error(f"Backend error: {resp.json().get('error', resp.text)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach backend at {BACKEND_URL}: {e}")

# batch prediction CSV file upload
with tab2:
    st.subheader("Upload a CSV for batch prediction")
    st.caption(
        "CSV must contain: Product_Weight, Product_Sugar_Content, Product_Allocated_Area, "
        "Product_MRP, Store_Size, Store_Location_City_Type, Store_Type, Product_Id_char, "
        "Store_Age_Years, Product_Type_Category"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        preview_df = pd.read_csv(uploaded_file)
        st.write("Preview:", preview_df.head())

        if st.button("Run Batch Prediction", type="primary"):
            uploaded_file.seek(0)
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/predict_batch",
                    files={"file": ("batch.csv", uploaded_file, "text/csv")},
                    timeout=30,
                )
                if resp.status_code == 200:
                    results = pd.DataFrame(resp.json()["predictions"])
                    st.success(f"Predicted {len(results)} rows.")
                    st.dataframe(results)
                    st.download_button(
                        "Download Predictions as CSV",
                        results.to_csv(index=False).encode("utf-8"),
                        file_name="superkart_batch_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"Backend error: {resp.json().get('error', resp.text)}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach backend at {BACKEND_URL}: {e}")
