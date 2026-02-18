
"""
This module defines a Streamlit GUI for batch scoring news headlines.
It allows users to paste a list of headlines, sends them to a local API,
and displays the sentiment results in a table.
"""
import logging
import requests
import pandas as pd
import streamlit as st

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
API_URL = "http://95.216.216.139:8086/score_headlines"

def main():
    """
    Main function to render the Streamlit interface and handle API logic.
    """
    st.set_page_config(page_title="Headline Analyzer", page_icon="📊")
    st.title("📊 Bulk Headline Analyzer")

    st.markdown("""
    **Instructions:**
    1. Paste your headlines below (one per line).
    2. Click 'Analyze Sentiments' to score them all.
    """)

    # Input Area
    text_input = st.text_area(
        "Paste Headlines Here",
        height=200,
        value="Market hits record high\nUnemployment rises slightly\nNew tech startup fails"
    )

    # Action Button
    if st.button("🚀 Analyze Sentiments"):
        logger.info("User clicked 'Analyze Sentiments'")

        # Process input: Split by newlines and remove empty strings
        lines = text_input.split('\n')
        valid_headlines = [line.strip() for line in lines if line.strip()]

        if not valid_headlines:
            st.warning("Please enter at least one headline.")
            logger.warning("User attempted to score empty input.")
            return

        logger.info("Sending %d headlines to API...", len(valid_headlines))

        with st.spinner(f"Scoring {len(valid_headlines)} headlines..."):
            try:
                # Prepare payload
                payload = {"headlines": valid_headlines}

                # API Call
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()

                # Parse Response
                data = response.json()
                labels = data.get("labels", [])

                logger.info("Received %d labels from API.", len(labels))

                # Create results DataFrame
                results_df = pd.DataFrame({
                    "Headline": valid_headlines,
                    "Prediction": labels
                })

                st.success("Analysis Complete!")

                # Display results
                # st.dataframe is compatible with older Streamlit versions
                st.dataframe(results_df)

            except requests.exceptions.RequestException as req_err:
                st.error("Connection Error: Could not reach the server.")
                logger.error("API Request failed: %s", req_err)

            except Exception as error:  # pylint: disable=broad-except
                # Catch-all for unexpected UI errors
                st.error(f"An unexpected error occurred: {error}")
                logger.error("Unexpected error: %s", error)

if __name__ == "__main__":
    main()
