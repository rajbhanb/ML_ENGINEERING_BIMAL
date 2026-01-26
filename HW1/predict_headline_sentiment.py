"""
This module processes a text file of headlines and predicts their sentiment
(Optimistic, Pessimistic, Neutral) using a pre-trained SVM model.
"""

import argparse
from datetime import datetime
import sys
import joblib
from sentence_transformers import SentenceTransformer


def parse_arguments():
    """
    Sets up argument parsing and returns the parsed arguments.
    Separating this reduces the local variable count in main().
    """
    parser = argparse.ArgumentParser(
        description="Use news headlines to predict sentiment using a pre-trained SVM."
    )

    # Add arguments (Strings split to satisfy line length limits)
    parser.add_argument(
        "Txt_file",
        type=str,
        help=("The full path to the text file with one headline per line. "
              "Be sure to put double apostrophe around the path.")
    )
    parser.add_argument(
        "Source",
        type=str,
        help=("The source of the headlines (e.g., New York Times). "
              "Be sure to put double apostrophe around the source.")
    )

    return parser.parse_args()


def main():
    """
    Main function to load models and process headlines.
    """
    # 1. Get Arguments
    args = parse_arguments()

    # 2. Load the model and classifier
    print("Loading models...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        clf = joblib.load('svm.joblib')
    except OSError as err:
        print(f"Error loading models (svm.joblib might be missing): {err}")
        sys.exit(1)

    # 3. Setup Filenames
    source_formatted = args.Source.replace(" ", "_")
    today = datetime.now().strftime("%Y_%m_%d")
    output_filename = f"headline_scores_{source_formatted}_{today}.txt"

    print(f"Reading from {args.Txt_file}...")

    # 4. Process the headlines
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

                # Predict sentiment
                prediction = clf.predict([embedding])[0]

                # Write format: Output, Original Headline
                outfile.write(f"{prediction}, {headline}\n")

        print(f"Success! Results saved to: {output_filename}")

    except FileNotFoundError:
        print(f"Error: Could not find '{args.Txt_file}'. "
              "Make sure the text file is in the specified directory.")

    except Exception as error:  # pylint: disable=broad-except
        print(f"An unexpected error occurred: {error}")


if __name__ == "__main__":
    main()
