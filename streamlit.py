import streamlit as st
import requests


# ==================================================
# FastAPI URL
# ==================================================

API_URL = "http://127.0.0.1:8000"


# ==================================================
# Page configuration
# ==================================================

st.set_page_config(
    page_title="Asian Paints Stock Prediction",
    page_icon="📈",
    layout="centered"
)


# ==================================================
# Title
# ==================================================

st.title("📈 Asian Paints Stock Prediction")

st.write(
    "AI-based prediction of Asian Paints stock "
    )

st.divider()


# ==================================================
# API Health Check
# ==================================================

st.subheader("API Status")

try:

    health_response = requests.get(
        f"{API_URL}/health",
        timeout=10
    )

    if health_response.status_code == 200:

        st.success("FastAPI is running")

    else:

        st.error("FastAPI is not healthy")

except requests.exceptions.ConnectionError:

    st.error(
        "Cannot connect to FastAPI. "
        "Please start the FastAPI server."
    )


st.divider()


# ==================================================
# Prediction Button
# ==================================================

st.subheader("Stock Prediction")


if st.button(
    "🔮 Predict Asian Paints",
    use_container_width=True
):

    with st.spinner(
        "Collecting market data and news..."
    ):

        try:

            response = requests.get(
                f"{API_URL}/predict",
                timeout=300
            )


            # ==========================================
            # Successful response
            # ==========================================

            if response.status_code == 200:

                result = response.json()


                if result["success"]:

                    data = result["data"]


                    # ==================================
                    # Get prediction values
                    # ==================================

                    prediction = data["prediction"]

                    up_probability = (
                        data["up_probability"]
                    )

                    down_probability = (
                        data["down_probability"]
                    )

                    prediction_date = (
                        data["date"]
                    )


                    # ==================================
                    # Display date
                    # ==================================

                    st.subheader(
                        f"Prediction Date: {prediction_date}"
                    )


                    # ==================================
                    # Display prediction
                    # ==================================

                    if prediction == "UP":

                        st.success(
                            "📈 Prediction: UP"
                        )

                    else:

                        st.error(
                            "📉 Prediction: DOWN"
                        )


                    # ==================================
                    # Probability columns
                    # ==================================

                    col1, col2 = st.columns(2)


                    with col1:

                        st.metric(
                            "UP Probability",
                            f"{up_probability * 100:.2f}%"
                        )


                    with col2:

                        st.metric(
                            "DOWN Probability",
                            f"{down_probability * 100:.2f}%"
                        )


                    # ==================================
                    # Confidence
                    # ==================================

                    st.subheader(
                        "Prediction Confidence"
                    )

                    st.progress(
                        up_probability
                    )


                    # ==================================
                    # Raw API response
                    # ==================================

                    with st.expander(
                        "View API Response"
                    ):

                        st.json(data)


                else:

                    st.error(
                        result.get(
                            "error",
                            "Prediction failed."
                        )
                    )


            # ==========================================
            # API returned an error
            # ==========================================

            else:

                try:

                    error_data = (
                        response.json()
                    )

                    st.error(
                        error_data.get(
                            "error",
                            "FastAPI returned an error."
                        )
                    )

                except Exception:

                    st.error(
                        f"FastAPI returned "
                        f"status code "
                        f"{response.status_code}"
                    )


        # ==============================================
        # Connection error
        # ==============================================

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI."
            )

            st.info(
                "Make sure you are running:\n\n"
                "`uvicorn app:app --reload`"
            )


        # ==============================================
        # Timeout
        # ==============================================

        except requests.exceptions.Timeout:

            st.error(
                "Prediction request timed out."
            )


        # ==============================================
        # Other errors
        # ==============================================

        except Exception as e:

            st.error(
                f"Unexpected error: {str(e)}"
            )
