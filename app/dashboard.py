import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from report_generator import generate_investment_report
import os
import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

# Add project root and src folder to Python path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_PATH))

# ============================================================
# IMPORTS
# ============================================================

from data_updater import update_all_companies

from report_generator import generate_investment_report

# ============================================================
# AUTOMATIC STOCK DATA UPDATE
# ============================================================

if "data_updated" not in st.session_state:

    with st.spinner("🔄 Updating latest market data..."):

        try:
            update_all_companies()

            st.session_state["data_updated"] = True

            st.success(
                "✅ Latest stock market data updated successfully."
            )

        except Exception as e:

            st.warning(
                f"⚠️ Could not update market data: {e}"
            )

# ============================================================
# PAGE STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "company_growth"
    / "final_company_growth_model.pkl"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "ml"
    / "company_growth_ml_dataset.csv"
)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load(MODEL_FILE)

df = pd.read_csv(DATA_FILE)


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Financial Intelligence",
    page_icon="🤖",
    layout="wide"
)
# ============================================================
# CUSTOM UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    /* Cards */
    .info-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 15px;
    }

    /* Section spacing */
    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 25px;
        opacity: 0.6;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    "## 🤖 AI Financial Intelligence"
)

st.sidebar.caption(
    "Financial Analysis Platform"
)

st.sidebar.divider()

pages = [
    "Dashboard",
    "Stock Prediction",
    "Company Growth",
    "Risk Analysis",
    "Investment Recommendation"
]

page = st.sidebar.radio(
    "Go to",
    pages,
    index=pages.index(st.session_state["page"])
)

st.session_state["page"] = page


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    # ============================================================
    # DASHBOARD HOME
    # ============================================================

    st.markdown(
    '<div class="main-title">🤖 AI Financial Intelligence</div>',
    unsafe_allow_html=True
)

    st.markdown(
    '<div class="subtitle">'
    'Smart financial analysis powered by Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

    st.write(
        "Analyze stocks, predict company growth, evaluate risk, "
        "and generate investment insights from one dashboard."
    )

    st.divider()

    # ============================================================
    # PROJECT MODULES
    # ============================================================

    st.subheader("🚀 Available Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            ### 📈 Stock Price Prediction

            Predict the next-day stock price using
            machine learning and technical indicators.

            **Model:** Linear Regression
            """
        )

        if st.button(
            "Open Stock Prediction",
            use_container_width=True
        ):
            st.session_state["page"] = "Stock Prediction"
            st.rerun()

    with col2:

        st.info(
            """
            ### 🏢 Company Growth

            Predict next-year revenue growth using
            company financial information.

            **Model:** Random Forest
            """
        )

        if st.button(
            "Open Company Growth",
            use_container_width=True
        ):
            st.session_state["page"] = "Company Growth"
            st.rerun()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.warning(
            """
            ### ⚠️ Risk Analysis

            Analyze volatility, RSI, moving averages
            and overall market risk.
            """
        )

        if st.button(
            "Open Risk Analysis",
            use_container_width=True
        ):
            st.session_state["page"] = "Risk Analysis"
            st.rerun()

    with col2:

        st.success(
            """
            ### 💰 Investment Recommendation

            Combine growth, market trend and risk
            information to generate an investment signal.
            """
        )

        if st.button(
            "Open Investment Analysis",
            use_container_width=True
        ):
            st.session_state["page"] = "Investment Recommendation"
            st.rerun()

    st.divider()

    # ============================================================
    # PROJECT PIPELINE
    # ============================================================

    st.subheader("🔄 AI Analysis Pipeline")

    st.markdown(
        """
        **Stock Data**
        ↓
        **Feature Engineering**
        ↓
        **Machine Learning Models**
        ↓
        **Stock Prediction**
        +
        **Company Growth Prediction**
        ↓
        **Risk Analysis**
        ↓
        **Investment Recommendation**
        """
    )

    st.divider()

    # ============================================================
    # TECHNOLOGIES
    # ============================================================

    st.subheader("🛠️ Technologies Used")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Language", "Python")

    with col2:
        st.metric("ML", "Scikit-learn")

    with col3:
        st.metric("Data", "Pandas")

    with col4:
        st.metric("Interface", "Streamlit")

    st.divider()

    # ============================================================
    # DISCLAIMER
    # ============================================================

    st.caption(
        "⚠️ This application is an educational machine-learning "
        "project. Predictions are estimates and should not be "
        "considered financial advice."
    )

# ============================================================
# STOCK PREDICTION
# ============================================================

elif page == "Stock Prediction":

    st.header("📈 Stock Price Prediction")

    st.write(
        "Predict the next-day stock price using the trained machine learning model."
    )

    # ============================================================
    # PROJECT PATH
    # ============================================================

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    MODEL_DIR = (
        PROJECT_ROOT
        / "models"
        / "stock_forecasting"
    )

    DATA_DIR = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "stock"
    )

    # ============================================================
    # AVAILABLE COMPANIES
    # ============================================================

    companies = {
        "Bharti Airtel": "Bharti_Airtel",
        "Coforge": "Coforge",
        "HCLTech": "HCLTech",
        "Hindustan Unilever": "Hindustan_Unilever",
        "Infosys": "Infosys",
        "KPIT Technologies": "KPIT_Technologies",
        "Larsen & Toubro": "Larsen_&_Toubro",
        "Mphasis": "Mphasis",
        "Persistent Systems": "Persistent_Systems",
        "Reliance Industries": "Reliance_Industries",
        "Tata Elxsi": "Tata_Elxsi",
        "TCS": "TCS",
        "Tech Mahindra": "Tech_Mahindra",
        "Wipro": "Wipro"
    }

    # ============================================================
    # COMPANY SELECTION
    # ============================================================

    st.subheader("🏢 Select Company")

    selected_company = st.selectbox(
        "Choose a company",
        list(companies.keys())
    )

    company_code = companies[selected_company]

    # ============================================================
    # MODEL + DATA PATH
    # ============================================================

    MODEL_FILE = (
        MODEL_DIR
        / f"{company_code}_random_forest.pkl"
    )

    DATA_FILE = (
        DATA_DIR
        / f"{company_code}_features.csv"
    )

    # ============================================================
    # DISPLAY SELECTED COMPANY
    # ============================================================

    st.info(
        f"📊 Analysis selected for **{selected_company}**"
    )

    # ============================================================
    # CHECK FILES
    # ============================================================

    if not MODEL_FILE.exists():

        st.error(
            f"❌ Model not found for {selected_company}"
        )

        st.write(
            f"Expected model: {MODEL_FILE}"
        )

    elif not DATA_FILE.exists():

        st.error(
            f"❌ Feature dataset not found for {selected_company}"
        )

        st.write(
            f"Expected dataset: {DATA_FILE}"
        )

    else:

        try:

            # ====================================================
            # LOAD MODEL
            # ====================================================

            model = joblib.load(
                MODEL_FILE
            )

            # ====================================================
            # LOAD DATA
            # ====================================================

            df = pd.read_csv(
                DATA_FILE
            )

            # ====================================================
            # DATE PROCESSING
            # ====================================================

            df["Date"] = pd.to_datetime(
                df["Date"]
            )

            df = (
                df
                .sort_values("Date")
                .reset_index(drop=True)
            )

            # ====================================================
            # FEATURES
            # ====================================================

            features = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Daily_Return",
                "Return_5D",
                "Return_20D",
                "SMA_20",
                "SMA_50",
                "EMA_20",
                "RSI_14",
                "MACD",
                "MACD_Signal",
                "MACD_Histogram",
                "Volatility_20",
                "Volume_Change",
                "Price_vs_SMA20",
                "Price_vs_SMA50"
            ]

            # ====================================================
            # CHECK FEATURES
            # ====================================================

            missing_features = [
                feature
                for feature in features
                if feature not in df.columns
            ]

            if missing_features:

                st.error(
                    "Some required features are missing."
                )

                st.write(
                    missing_features
                )

            else:

                # =================================================
                # LATEST DATA
                # =================================================

                latest = df.iloc[-1]

                latest_date = latest["Date"]

                current_price = float(
                    latest["Close"]
                )

                # =================================================
                # PREPARE INPUT
                # =================================================

                input_data = (
                    df[features]
                    .iloc[[-1]]
                    .copy()
                )

                # =================================================
                # PREDICTION
                # =================================================

                prediction = float(
                    model.predict(input_data)[0]
                )

                # =================================================
                # LATEST STOCK INFORMATION
                # =================================================

                st.divider()

                st.subheader(
                    "📊 Latest Stock Information"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Company",
                        selected_company
                    )

                with col2:

                    st.metric(
                        "Latest Date",
                        latest_date.strftime(
                            "%Y-%m-%d"
                        )
                    )

                with col3:

                    st.metric(
                        "Current Price",
                        f"₹{current_price:,.2f}"
                    )

                # =================================================
                # PREDICTION RESULT
                # =================================================

                st.divider()

                st.subheader(
                    "🔮 Next-Day Prediction"
                )

                col1, col2, col3 = st.columns(3)

                price_change = (
                    prediction - current_price
                )

                percentage_change = (
                    price_change / current_price
                ) * 100

                with col1:

                    st.metric(
                        "Current Price",
                        f"₹{current_price:,.2f}"
                    )

                with col2:

                    st.metric(
                        "Predicted Price",
                        f"₹{prediction:,.2f}"
                    )

                with col3:

                    st.metric(
                        "Expected Change",
                        f"{percentage_change:+.2f}%"
                    )

                # =================================================
                # OUTLOOK
                # =================================================

                if prediction > current_price:

                    st.success(
                        f"📈 Potential UPWARD movement "
                        f"of ₹{price_change:,.2f}"
                    )

                elif prediction < current_price:

                    st.warning(
                        f"📉 Potential DOWNWARD movement "
                        f"of ₹{abs(price_change):,.2f}"
                    )

                else:

                    st.info(
                        "➡️ NEUTRAL movement"
                    )

                # =================================================
                # PRICE CHART
                # =================================================

                st.divider()

                st.subheader(
                    f"📈 {selected_company} Price History"
                )

                chart_data = (
                    df
                    .set_index("Date")["Close"]
                )

                st.line_chart(
                    chart_data
                )

                # =================================================
                # MODEL INFORMATION
                # =================================================

                with st.expander(
                    "🔍 View Features Used"
                ):

                    feature_data = (
                        input_data.T
                    )

                    feature_data.columns = [
                        "Latest Value"
                    ]

                    st.dataframe(
                        feature_data,
                        use_container_width=True
                    )

                # =================================================
                # FILE INFORMATION
                # =================================================

                with st.expander(
                    "📁 Model Information"
                ):

                    st.write(
                        f"**Company:** {selected_company}"
                    )

                    st.write(
                        f"**Model:** Random Forest"
                    )

                    st.write(
                        f"**Model File:** "
                        f"`{MODEL_FILE.name}`"
                    )

                    st.write(
                        f"**Dataset:** "
                        f"`{DATA_FILE.name}`"
                    )

        except Exception as e:

            st.error(
                "❌ Error while generating prediction."
            )

            st.exception(e)
# ============================================================
# COMPANY GROWTH
# ============================================================

elif page == "Company Growth":

    st.header("🏢 Company Growth Prediction")

    st.write(
        "Predict next-year revenue growth using machine learning."
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = [
        "Revenue_Growth",
        "Profit_Growth",
        "EPS_Growth",
        "EBIT_Margin",
        "Net_Profit_Margin",
        "Revenue_Growth_Change",
        "Profit_Growth_Change",
        "EPS_Growth_Change"
    ]

    # --------------------------------------------------------
    # COMPANY SELECTION
    # --------------------------------------------------------

    companies = sorted(
        df["Company"].dropna().unique()
    )

    company = st.selectbox(
        "Select Company",
        companies
    )

    # --------------------------------------------------------
    # YEAR SELECTION
    # --------------------------------------------------------

    years = sorted(
        df["Year"].dropna().unique(),
        reverse=True
    )

    year = st.selectbox(
        "Select Financial Year",
        years
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Company Growth",
        use_container_width=True
    ):

        company_data = df[
            (df["Company"] == company) &
            (df["Year"] == year)
        ]

        if company_data.empty:

            st.error(
                "No data available for this company and year."
            )

        else:

            input_data = company_data[features]

            prediction = model.predict(
                input_data
            )[0]

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "📊 Prediction Result"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Company",
                    company
                )

            with col2:

                st.metric(
                    "Financial Year",
                    year
                )

            with col3:

                st.metric(
                    "Predicted Growth",
                    f"{prediction:.2f}%"
                )

            # ------------------------------------------------
            # OUTLOOK
            # ------------------------------------------------

            if prediction >= 15:

                outlook = "HIGH GROWTH"

            elif prediction >= 8:

                outlook = "MODERATE GROWTH"

            elif prediction >= 0:

                outlook = "LOW GROWTH"

            else:

                outlook = "NEGATIVE GROWTH"

            st.subheader(
                "Growth Outlook"
            )

            if prediction >= 15:

                st.success(outlook)

            elif prediction >= 8:

                st.info(outlook)

            elif prediction >= 0:

                st.warning(outlook)

            else:

                st.error(outlook)

            # ------------------------------------------------
            # FEATURES USED
            # ------------------------------------------------

            st.subheader(
                "Financial Features Used"
            )

            feature_display = company_data[features].T

            feature_display.columns = ["Value"]

            st.dataframe(
                feature_display,
                use_container_width=True
            )

# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.header("⚠️ Risk Analysis")

    st.write(
        "Analyze stock volatility, RSI, moving averages and overall market risk."
    )

    # ============================================================
    # PROJECT PATH
    # ============================================================

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    STOCK_DATA_DIR = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "stock"
    )

    # ============================================================
    # COMPANY FILE MAPPING
    # ============================================================

    company_files = {
        "TCS": "TCS_features.csv",
        "Infosys": "Infosys_features.csv",
        "Wipro": "Wipro_features.csv",
        "HCLTech": "HCLTech_features.csv",
        "Tech Mahindra": "Tech_Mahindra_features.csv",
        "Persistent Systems": "Persistent_Systems_features.csv",
        "Mphasis": "Mphasis_features.csv",
        "Coforge": "Coforge_features.csv",
        "Larsen & Toubro": "Larsen_&_Toubro_features.csv",
        "Reliance Industries": "Reliance_Industries_features.csv",
        "Hindustan Unilever": "Hindustan_Unilever_features.csv",
        "Bharti Airtel": "Bharti_Airtel_features.csv",
        "KPIT Technologies": "KPIT_Technologies_features.csv",
        "Tata Elxsi": "Tata_Elxsi_features.csv",
    }

    # ============================================================
    # COMPANY SELECTION
    # ============================================================

    st.subheader("🏢 Select Company")

    selected_company = st.selectbox(
        "Choose a company for risk analysis:",
        list(company_files.keys())
    )

    data_file = STOCK_DATA_DIR / company_files[selected_company]

    # ============================================================
    # CHECK FILE
    # ============================================================

    if not data_file.exists():

        st.error(
            f"Risk analysis data not found for {selected_company}."
        )

        st.write("Expected file:")
        st.code(str(data_file))

    else:

        # ========================================================
        # LOAD DATA
        # ========================================================

        df = pd.read_csv(data_file)

        df["Date"] = pd.to_datetime(df["Date"])

        df = (
            df
            .sort_values("Date")
            .reset_index(drop=True)
        )

        # ========================================================
        # LATEST DATA
        # ========================================================

        latest = df.iloc[-1]

        latest_date = latest["Date"]

        closing_price = latest["Close"]

        # ========================================================
        # GET RISK FEATURES
        # ========================================================

        volatility = (
            latest["Volatility_20"]
            if "Volatility_20" in df.columns
            else 0
        )

        rsi = (
            latest["RSI_14"]
            if "RSI_14" in df.columns
            else 50
        )

        sma20 = (
            latest["SMA_20"]
            if "SMA_20" in df.columns
            else closing_price
        )

        sma50 = (
            latest["SMA_50"]
            if "SMA_50" in df.columns
            else closing_price
        )

        # ========================================================
        # OVERALL RISK
        # ========================================================

        if volatility >= 0.04:
            risk_level = "HIGH"
        elif volatility >= 0.02:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # RSI adjustment

        if rsi >= 70 or rsi <= 30:
            if risk_level == "LOW":
                risk_level = "MODERATE"

        # ========================================================
        # HEADER
        # ========================================================

        st.divider()

        st.subheader("📊 Current Market Data")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Company",
                selected_company
            )

        with col2:

            st.metric(
                "Latest Date",
                latest_date.strftime("%Y-%m-%d")
            )

        with col3:

            st.metric(
                "Closing Price",
                f"₹{closing_price:.2f}"
            )

        # ========================================================
        # RISK METRICS
        # ========================================================

        st.divider()

        st.subheader("📈 Risk Indicators")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "20-Day Volatility",
                f"{volatility:.4f}"
            )

        with col2:

            st.metric(
                "RSI (14)",
                f"{rsi:.2f}"
            )

        with col3:

            st.metric(
                "Overall Risk",
                risk_level
            )

        # ========================================================
        # RISK MESSAGE
        # ========================================================

        if risk_level == "HIGH":

            st.error(
                "🔴 High Risk: The stock is showing relatively high volatility."
            )

        elif risk_level == "MODERATE":

            st.warning(
                "🟡 Moderate Risk: The stock has noticeable market volatility."
            )

        else:

            st.success(
                "🟢 Low Risk: The stock is currently showing relatively low volatility."
            )

        # ========================================================
        # MOVING AVERAGE ANALYSIS
        # ========================================================

        st.divider()

        st.subheader("📉 Moving Average Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "SMA 20",
                f"₹{sma20:.2f}"
            )

        with col2:

            st.metric(
                "SMA 50",
                f"₹{sma50:.2f}"
            )

        if closing_price > sma20 and closing_price > sma50:

            st.success(
                "📈 Price is above both SMA 20 and SMA 50."
            )

        elif closing_price < sma20 and closing_price < sma50:

            st.warning(
                "📉 Price is below both SMA 20 and SMA 50."
            )

        else:

            st.info(
                "➡️ Price is between the moving averages."
            )

        # ========================================================
        # RSI ANALYSIS
        # ========================================================

        st.divider()

        st.subheader("📊 RSI Analysis")

        if rsi >= 70:

            st.warning(
                f"⚠️ RSI = {rsi:.2f} — potentially overbought."
            )

        elif rsi <= 30:

            st.warning(
                f"⚠️ RSI = {rsi:.2f} — potentially oversold."
            )

        else:

            st.success(
                f"✅ RSI = {rsi:.2f} — within a normal range."
            )

        # ========================================================
        # PRICE HISTORY
        # ========================================================

        st.divider()

        st.subheader(
            f"📈 {selected_company} Price History"
        )

        chart_data = (
            df
            .set_index("Date")["Close"]
        )

        st.line_chart(chart_data)

        # ========================================================
        # RISK DATA
        # ========================================================

        with st.expander("🔍 View Risk Data"):

            risk_data = pd.DataFrame({
                "Indicator": [
                    "Closing Price",
                    "20-Day Volatility",
                    "RSI (14)",
                    "SMA 20",
                    "SMA 50",
                    "Overall Risk"
                ],
                "Value": [
                    f"₹{closing_price:.2f}",
                    f"{volatility:.4f}",
                    f"{rsi:.2f}",
                    f"₹{sma20:.2f}",
                    f"₹{sma50:.2f}",
                    risk_level
                ]
            })

            st.dataframe(
                risk_data,
                use_container_width=True,
                hide_index=True
            )

# ============================================================
# INVESTMENT RECOMMENDATION
# ============================================================

elif page == "Investment Recommendation":

    st.header("💰 Investment Recommendation")

    from report_generator import generate_investment_report

    st.write(
        "AI-based investment analysis combining stock prediction, "
        "market trend and risk indicators."
    )

    # ============================================================
    # PROJECT PATH
    # ============================================================

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    STOCK_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "stock"

    MODEL_DIR = PROJECT_ROOT / "models" / "stock_forecasting"

    # ============================================================
    # COMPANY FILE MAPPING
    # ============================================================

    company_files = {
        "TCS": "TCS_features.csv",
        "Infosys": "Infosys_features.csv",
        "Wipro": "Wipro_features.csv",
        "HCLTech": "HCLTech_features.csv",
        "Tech Mahindra": "Tech_Mahindra_features.csv",
        "Persistent Systems": "Persistent_Systems_features.csv",
        "Mphasis": "Mphasis_features.csv",
        "Coforge": "Coforge_features.csv",
        "Larsen & Toubro": "Larsen_&_Toubro_features.csv",
        "Reliance Industries": "Reliance_Industries_features.csv",
        "Hindustan Unilever": "Hindustan_Unilever_features.csv",
        "Bharti Airtel": "Bharti_Airtel_features.csv",
        "KPIT Technologies": "KPIT_Technologies_features.csv",
        "Tata Elxsi": "Tata_Elxsi_features.csv",
    }

    # ============================================================
    # COMPANY SELECTION
    # ============================================================

    st.subheader("🏢 Select Company")

    selected_company = st.selectbox(
        "Choose a company for investment analysis:",
        list(company_files.keys())
    )

    # ============================================================
    # FILE PATHS
    # ============================================================

    data_filename = company_files[selected_company]

    company_code = data_filename.replace("_features.csv", "")

    data_file = STOCK_DATA_DIR / data_filename

    model_file = MODEL_DIR / f"{company_code}_linear_regression.pkl"

    # ============================================================
    # CHECK FILES
    # ============================================================

    if not data_file.exists():

        st.error(f"❌ Data file not found for {selected_company}.")
        st.code(str(data_file))

    elif not model_file.exists():

        st.error(f"❌ Prediction model not found for {selected_company}.")
        st.code(str(model_file))

    else:

        try:

            # ====================================================
            # LOAD DATA
            # ====================================================

            df = pd.read_csv(data_file)

            df["Date"] = pd.to_datetime(df["Date"])

            df = (
                df
                .sort_values("Date")
                .reset_index(drop=True)
            )

            # ====================================================
            # FEATURES
            # ====================================================

            features = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Daily_Return",
                "Return_5D",
                "Return_20D",
                "SMA_20",
                "SMA_50",
                "EMA_20",
                "RSI_14",
                "MACD",
                "MACD_Signal",
                "MACD_Histogram",
                "Volatility_20",
                "Volume_Change",
                "Price_vs_SMA20",
                "Price_vs_SMA50",
            ]

            # ====================================================
            # CHECK FEATURES
            # ====================================================

            missing_features = [
                feature
                for feature in features
                if feature not in df.columns
            ]

            if missing_features:

                st.error("❌ Required features are missing.")
                st.write(missing_features)

            else:

                # =================================================
                # LOAD MODEL
                # =================================================

                model = joblib.load(model_file)

                # =================================================
                # LATEST DATA
                # =================================================

                latest = df.iloc[-1]

                current_price = float(latest["Close"])

                latest_date = latest["Date"]

                # =================================================
                # PREPARE MODEL INPUT
                # =================================================

                input_data = (
                    df[features]
                    .iloc[[-1]]
                    .copy()
                )

                # =================================================
                # CHECK MISSING VALUES
                # =================================================

                if input_data.isnull().any().any():

                    st.error(
                        "❌ Latest financial data contains missing values."
                    )

                else:

                    # =============================================
                    # PREDICTION
                    # =============================================

                    prediction = float(
                        model.predict(input_data)[0]
                    )

                    # =============================================
                    # EXPECTED PRICE CHANGE
                    # =============================================

                    price_change = prediction - current_price

                    expected_growth = (
                        price_change / current_price
                    ) * 100

                    # =============================================
                    # RSI
                    # =============================================

                    rsi = float(latest["RSI_14"])

                    # =============================================
                    # VOLATILITY
                    # =============================================

                    volatility = float(latest["Volatility_20"])

                    # =============================================
                    # MOVING AVERAGES
                    # =============================================

                    sma20 = float(latest["SMA_20"])

                    sma50 = float(latest["SMA_50"])

                    # =============================================
                    # MARKET TREND
                    # =============================================

                    if (
                        current_price > sma20
                        and sma20 > sma50
                    ):

                        market_trend = "BULLISH"

                    elif (
                        current_price < sma20
                        and sma20 < sma50
                    ):

                        market_trend = "BEARISH"

                    else:

                        market_trend = "NEUTRAL"

                    # =============================================
                    # RISK LEVEL
                    # =============================================

                    if volatility >= 0.04:

                        risk_level = "HIGH"

                    elif volatility >= 0.02:

                        risk_level = "MODERATE"

                    else:

                        risk_level = "LOW"

                    # =============================================
                    # RECOMMENDATION SCORE
                    # =============================================

                    score = 0

                    # Expected Growth
                    if expected_growth >= 2:

                        score += 2

                    elif expected_growth > 0:

                        score += 1

                    else:

                        score -= 1

                    # Market Trend
                    if market_trend == "BULLISH":

                        score += 2

                    elif market_trend == "BEARISH":

                        score -= 2

                    # Risk
                    if risk_level == "LOW":

                        score += 1

                    elif risk_level == "HIGH":

                        score -= 2

                    # RSI
                    if 40 <= rsi <= 65:

                        score += 1

                    elif rsi >= 70:

                        score -= 1

                    elif rsi <= 30:

                        score -= 1

                    # =================================================
                    # FINAL RECOMMENDATION
                    # =================================================

                    if score >= 4:

                        recommendation = "BUY"
                        confidence = "HIGH"

                    elif score >= 1:

                        recommendation = "HOLD"
                        confidence = "MODERATE"

                    else:

                        recommendation = "AVOID"
                        confidence = "MODERATE"

                    # =================================================
                    # COMPANY INFORMATION
                    # =================================================

                    st.divider()

                    st.subheader("🏢 Company Analysis")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Company", selected_company)

                    with col2:
                        st.metric(
                            "Latest Date",
                            latest_date.strftime("%Y-%m-%d")
                        )

                    with col3:
                        st.metric(
                            "Current Price",
                            f"₹{current_price:,.2f}"
                        )

                    # =================================================
                    # PRICE PREDICTION
                    # =================================================

                    st.divider()

                    st.subheader("🔮 Stock Prediction")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Current Price",
                            f"₹{current_price:,.2f}"
                        )

                    with col2:
                        st.metric(
                            "Predicted Price",
                            f"₹{prediction:,.2f}"
                        )

                    with col3:
                        st.metric(
                            "Expected Growth",
                            f"{expected_growth:+.2f}%"
                        )

                    # =================================================
                    # MARKET SIGNALS
                    # =================================================

                    st.divider()

                    st.subheader("📊 Market Signals")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Market Trend", market_trend)

                    with col2:
                        st.metric("RSI (14)", f"{rsi:.2f}")

                    with col3:
                        st.metric("Risk Level", risk_level)

                    # =================================================
                    # FINAL RECOMMENDATION
                    # =================================================

                    st.divider()

                    st.subheader("🎯 Final Recommendation")

                    if recommendation == "BUY":

                        st.success(
                            f"""
                            ## 🟢 BUY

                            **Confidence: {confidence}**

                            The current analysis indicates a potentially
                            positive investment signal based on the model
                            prediction, market trend and risk indicators.
                            """
                        )

                    elif recommendation == "HOLD":

                        st.warning(
                            f"""
                            ## 🟡 HOLD

                            **Confidence: {confidence}**

                            The current analysis shows mixed or moderate
                            signals. Monitoring the stock may be appropriate.
                            """
                        )

                    else:

                        st.error(
                            f"""
                            ## 🔴 AVOID

                            **Confidence: {confidence}**

                            The current analysis indicates unfavorable
                            conditions based on predicted movement,
                            trend and risk.
                            """
                        )

                    # =================================================
                    # SCORE
                    # =================================================

                    with st.expander("🔍 View Recommendation Score"):

                        st.metric("Overall Score", score)

                        st.write("The score combines:")
                        st.write("• Expected price growth")
                        st.write("• Market trend")
                        st.write("• Volatility-based risk")
                        st.write("• RSI indicator")

                    # =================================================
                    # REASONS
                    # =================================================

                    st.divider()

                    st.subheader("🧠 Why this recommendation?")

                    reasons = []

                    if expected_growth > 0:
                        reasons.append(
                            f"The model predicts a {expected_growth:.2f}% "
                            f"increase in the next-day price."
                        )
                    else:
                        reasons.append(
                            f"The model predicts a {abs(expected_growth):.2f}% "
                            f"decrease in the next-day price."
                        )

                    if market_trend == "BULLISH":
                        reasons.append(
                            "The stock is showing a bullish moving-average trend."
                        )
                    elif market_trend == "BEARISH":
                        reasons.append(
                            "The stock is showing a bearish moving-average trend."
                        )
                    else:
                        reasons.append(
                            "The moving-average trend is currently neutral."
                        )

                    if risk_level == "LOW":
                        reasons.append(
                            "20-day volatility indicates relatively low risk."
                        )
                    elif risk_level == "MODERATE":
                        reasons.append(
                            "20-day volatility indicates moderate risk."
                        )
                    else:
                        reasons.append(
                            "20-day volatility indicates relatively high risk."
                        )

                    if rsi >= 70:
                        reasons.append(
                            f"RSI is {rsi:.2f}, indicating potentially "
                            "overbought conditions."
                        )
                    elif rsi <= 30:
                        reasons.append(
                            f"RSI is {rsi:.2f}, indicating potentially "
                            "oversold conditions."
                        )
                    else:
                        reasons.append(
                            f"RSI is {rsi:.2f}, which is not in an extreme zone."
                        )

                    for reason in reasons:
                        st.write("• " + reason)

                    # =================================================
                    # ANALYSIS SUMMARY
                    # =================================================

                    st.divider()

                    st.subheader("📋 Analysis Summary")

                    summary = pd.DataFrame({
                        "Factor": [
                            "Company",
                            "Current Price",
                            "Predicted Price",
                            "Expected Growth",
                            "Market Trend",
                            "RSI",
                            "20-Day Volatility",
                            "Risk Level",
                            "Recommendation",
                            "Confidence",
                            "Score",
                        ],
                        "Value": [
                            selected_company,
                            f"₹{current_price:,.2f}",
                            f"₹{prediction:,.2f}",
                            f"{expected_growth:+.2f}%",
                            market_trend,
                            f"{rsi:.2f}",
                            f"{volatility:.4f}",
                            risk_level,
                            recommendation,
                            confidence,
                            score,
                        ],
                    })

                    st.dataframe(
                        summary,
                        use_container_width=True,
                        hide_index=True
                    )

                    # =================================================
                    # PROFESSIONAL PDF REPORT
                    # =================================================

                    st.divider()

                    st.subheader("📄 Professional Investment Report")

                    st.write(
                        "Generate a portfolio-ready PDF containing all four "
                        "analysis sections, a recent price chart, model "
                        "interpretation, risk analysis, recommendation score, "
                        "AI-generated professional commentary and disclaimer."
                    )

                    pdf_file = generate_investment_report(
                        selected_company=selected_company,
                        latest_date=latest_date,
                        current_price=current_price,
                        prediction=prediction,
                        expected_growth=expected_growth,
                        rsi=rsi,
                        volatility=volatility,
                        sma20=sma20,
                        sma50=sma50,
                        market_trend=market_trend,
                        risk_level=risk_level,
                        score=score,
                        recommendation=recommendation,
                        confidence=confidence,
                        df=df,
                    )

                    st.download_button(
                        label="📥 Download Complete Investment Report",
                        data=pdf_file,
                        file_name=(
                            f"{company_code}_Investment_Report.pdf"
                        ),
                        mime="application/pdf",
                        use_container_width=True,
                    )

                    # =================================================
                    # DISCLAIMER
                    # =================================================

                    st.divider()

                    st.info(
                        "⚠️ Educational project only. This recommendation "
                        "is generated from historical market data and "
                        "machine-learning predictions. It is not financial "
                        "advice and should not be used as the sole basis "
                        "for investment decisions."
                    )

        except Exception as e:

            st.error(
                "❌ Error while generating investment recommendation."
            )

            st.exception(e)
