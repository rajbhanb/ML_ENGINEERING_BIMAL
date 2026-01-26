
import argparse
import joblib
from sentence_transformers import SentenceTransformer
# Load the model and classifier
model = SentenceTransformer("all-MiniLM-L6-v2")
clf = joblib.load('svm.joblib')

from datetime import datetime

# 1. Set up Argument Parser
parser = argparse.ArgumentParser(description="Use news headlines to predict sentiment using a pre-trained SVM.")

# Add arguments
parser.add_argument("Txt_file", type=str, help="The full path to the text file with one headline per line. Be sure to put double apostrophe around the path.")
parser.add_argument("Source", type=str, help="The source of the headlines (e.g., New York Times). Be sure to put double apostrophe around the source.")

# Parse the arguments
args = parser.parse_args()

# Replace spaces with underscores in the source name
source_formatted = args.Source.replace(" ", "_")

# Generate today's date in the specified format
today = datetime.now().strftime("%Y_%m_%d")

# Define output filename
output_filename = f"headline_scores_{source_formatted}_{today}.txt"

# Process the headlines and generate output
print(f"Reading from {args.Txt_file}...")

try:
    with open(args.Txt_file, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:

        lines = infile.readlines()

        for line in lines:
            headline = line.strip()

            # Skip empty lines
            if not headline:
                continue

            # Generate embedding
            embedding = model.encode(headline)

            # Predict sentiment (clf.predict expects a list or 2D array)
            prediction = clf.predict([embedding])[0]

            # Write format: Output, Original Headline
            outfile.write(f"{prediction}, {headline}\n")

    print(f"Success! Results saved to: {output_filename}")

except FileNotFoundError:
    print(f"Error: Could not find '{args.Txt_file}'. Make sure the text file is in the specified directory.")

except Exception as e:
    print(f"An error occurred: {e}")
