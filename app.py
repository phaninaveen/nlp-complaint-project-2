import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from wordcloud import WordCloud

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Complaint Theme Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# MODERN UI DESIGN
# ---------------------------------------------------

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #f8fafc, #e2e8f0);
}

/* Main Title */
.main-title {
    font-size: 45px;
    font-weight: bold;
    color: #0f172a;
    text-align: center;
    margin-top: 10px;
}

/* Subtitle */
.sub-title {
    font-size: 20px;
    color: #475569;
    text-align: center;
    margin-bottom: 30px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 2px solid #e2e8f0;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(to right, #3b82f6, #06b6d4);
    color: white;
    border-radius: 12px;
    height: 55px;
    width: 100%;
    border: none;
    font-size: 18px;
    font-weight: bold;
}

/* Text Area */
.stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #cbd5e1;
    padding: 10px;
}

/* Cards */
.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 20px;
}

/* Headings */
h2, h3 {
    color: #0f172a;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE SECTION
# ---------------------------------------------------

st.markdown(
    """
    <div class='main-title'>
    🧠 AI Complaint Theme Detection Dashboard
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='sub-title'>
    Detect hidden customer complaint themes using NLP and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# DASHBOARD CARDS
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='metric-box'>
    <h2>📄</h2>
    <h3>Complaint Analysis</h3>
    <p>Analyze customer issues instantly</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-box'>
    <h2>🧠</h2>
    <h3>AI Theme Detection</h3>
    <p>Automatically identify hidden topics</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-box'>
    <h2>📊</h2>
    <h3>Visual Insights</h3>
    <p>Interactive charts and analytics</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("⚙️ Dashboard Settings")

num_clusters = st.sidebar.slider(
    "Select Number of Themes",
    min_value=2,
    max_value=6,
    value=4
)

# ---------------------------------------------------
# SAMPLE DATA
# ---------------------------------------------------

sample_complaints = """
The delivery was delayed by five days.
Refund process is very slow and frustrating.
Customer support did not answer my calls.
The app crashes every time I open it.
Payment failed but money was deducted.
Product quality is poor and damaged.
Website loading speed is very slow.
The package arrived broken.
Support team resolved my issue quickly.
The mobile app interface is confusing.
Delivery tracking information is incorrect.
Received wrong item in the package.
Refund has not been credited yet.
Technical support is unhelpful.
The checkout page freezes frequently.
Product stopped working after one week.
Customer care executive was rude.
The order arrived earlier than expected.
Payment gateway has many bugs.
The application keeps logging me out.
"""

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.subheader("📥 Upload or Paste Complaint Logs")

input_method = st.radio(
    "Choose Input Method",
    ["Use Sample Data", "Paste Complaints", "Upload CSV"]
)

reviews = []

# SAMPLE DATA
if input_method == "Use Sample Data":

    complaint_text = st.text_area(
        "Complaint Logs",
        sample_complaints,
        height=300
    )

    reviews = [
        line.strip()
        for line in complaint_text.split("\n")
        if len(line.strip()) > 5
    ]

# PASTE COMPLAINTS
elif input_method == "Paste Complaints":

    complaint_text = st.text_area(
        "Paste Complaints Here",
        height=300
    )

    reviews = [
        line.strip()
        for line in complaint_text.split("\n")
        if len(line.strip()) > 5
    ]

# CSV UPLOAD
else:

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df_upload = pd.read_csv(uploaded_file)

        st.write("### Uploaded Dataset")
        st.dataframe(df_upload.head())

        column_name = st.selectbox(
            "Select Complaint Column",
            df_upload.columns
        )

        reviews = (
            df_upload[column_name]
            .dropna()
            .astype(str)
            .tolist()
        )

# ---------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------

if st.button("🚀 Analyze Complaints"):

    if len(reviews) < num_clusters:
        st.error("Not enough complaints for clustering.")
        st.stop()

    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000
    )

    X = vectorizer.fit_transform(reviews)

    # K-MEANS
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X)

    # DATAFRAME
    df = pd.DataFrame({
        "Complaint": reviews,
        "Cluster": clusters
    })

    # LABELS
    labels = {
        0: "Delivery Issues",
        1: "Refund Problems",
        2: "Technical Errors",
        3: "Customer Support",
        4: "Payment Issues",
        5: "Product Quality"
    }

    df["Theme"] = df["Cluster"].map(labels)

    # ---------------------------------------------------
    # RESULTS TABLE
    # ---------------------------------------------------

    st.subheader("📋 Clustered Complaints")
    st.dataframe(df)

    # ---------------------------------------------------
    # THEME DISTRIBUTION
    # ---------------------------------------------------

    cluster_counts = df["Theme"].value_counts()

    st.subheader("📈 Complaint Theme Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x=cluster_counts.index,
        y=cluster_counts.values,
        ax=ax
    )

    plt.xticks(rotation=15)

    st.pyplot(fig)

    # ---------------------------------------------------
    # PIE CHART
    # ---------------------------------------------------

    st.subheader("🥧 Complaint Share")

    fig2, ax2 = plt.subplots(figsize=(7, 7))

    ax2.pie(
        cluster_counts,
        labels=cluster_counts.index,
        autopct='%1.1f%%'
    )

    st.pyplot(fig2)

    # ---------------------------------------------------
    # PCA VISUALIZATION
    # ---------------------------------------------------

    st.subheader("🧩 Complaint Cluster Visualization")

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(X.toarray())

    pca_df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "Theme": df["Theme"]
    })

    fig3, ax3 = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=pca_df,
        x="x",
        y="y",
        hue="Theme",
        s=120
    )

    st.pyplot(fig3)

    # ---------------------------------------------------
    # WORD CLOUD
    # ---------------------------------------------------

    st.subheader("☁️ Common Complaint Keywords")

    all_text = " ".join(reviews)

    wordcloud = WordCloud(
        width=900,
        height=400,
        background_color='white',
        stopwords=ENGLISH_STOP_WORDS
    ).generate(all_text)

    fig4, ax4 = plt.subplots(figsize=(12, 5))

    ax4.imshow(wordcloud, interpolation='bilinear')
    ax4.axis('off')

    st.pyplot(fig4)

    # ---------------------------------------------------
    # TOP KEYWORDS
    # ---------------------------------------------------

    st.subheader("🔑 Top Keywords")

    terms = vectorizer.get_feature_names_out()

    sums = X.sum(axis=0)

    word_freq = [
        (word, sums[0, idx])
        for word, idx in vectorizer.vocabulary_.items()
    ]

    word_freq = sorted(
        word_freq,
        key=lambda x: x[1],
        reverse=True
    )[:10]

    keywords_df = pd.DataFrame(
        word_freq,
        columns=["Word", "Score"]
    )

    fig5, ax5 = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=keywords_df,
        x="Score",
        y="Word",
        ax=ax5
    )

    st.pyplot(fig5)

    # ---------------------------------------------------
    # INSIGHTS
    # ---------------------------------------------------

    st.subheader("📌 Insights Summary")

    most_common_theme = cluster_counts.idxmax()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success(
            f"Most Common Theme:\n{most_common_theme}"
        )

    with col2:
        st.info(
            f"Total Complaints:\n{len(reviews)}"
        )

    with col3:
        st.warning(
            f"Themes Detected:\n{num_clusters}"
        )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <center>
    Built using Streamlit, NLP, TF-IDF, and K-Means Clustering
    </center>
    """,
    unsafe_allow_html=True
)
