# TrainWise Exercises - AI Fitness Trainer

This project is an AI-powered fitness trainer that uses MediaPipe for pose estimation to analyze exercises like squats.

## 🚀 Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GreyNova/TrainWise_Exercises.git
    cd TrainWise_Exercises
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the app:**
    ```bash
    streamlit run Demo.py
    ```

## ☁️ Deploy to Streamlit Community Cloud

1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Connect your GitHub account.
3.  Click **"New app"**.
4.  Select the repository: `GreyNova/TrainWise_Exercises`.
5.  Select the branch: `main`.
6.  **Crucial Step:** In the "Main file path" field, enter `Demo.py` (instead of the default `streamlit_app.py`).
7.  Click **"Deploy"**.

## 🛠️ Tech Stack
-   **Python**
-   **Streamlit**
-   **MediaPipe**
-   **OpenCV**
-   **NumPy**
