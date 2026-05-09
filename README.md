# Afaan-Oromo IR Retrieval System 1

This is a comprehensive Information Retrieval (IR) and Natural Language Processing (NLP) system built for the **Afaan Oromo** language. It features a rule-based stemmer, a TF-IDF based search engine, dictionary lookups, and document management with file upload capabilities.

## Features

- 🔬 **Rule-Based Stemmer**: A context-sensitive stemmer that applies 7 rule clusters to handle Afaan Oromo morphology (suffixes, inflections, tense, voice, reduplication).
- 🔍 **Information Retrieval (TF-IDF)**: A robust search engine that indexes documents using TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity. Both queries and documents are stemmed before indexing to ensure morphology-aware retrieval.
- 📖 **COED Dictionary Lookup**: Integrated access to the Comprehensive Oromo-English Dictionary (COED) by Tilahun Gamta (2004), allowing you to look up headwords, their part-of-speech tags, and definitions.
- 📚 **Document Corpus Management**: View existing documents, add new text documents manually, or **upload files (`.txt`, `.pdf`, `.docx`)**. Uploaded files are automatically parsed, added to the corpus, and immediately indexed for searching.

## Requirements

Ensure you have Python 3 installed. This project requires the following Python libraries:

- `Flask` (for the web interface and API)
- `PyPDF2` (for parsing PDF document uploads)
- `python-docx` (for parsing Word Document uploads)
- `Werkzeug` (for secure file handling)

You can install all dependencies using the provided `requirements.txt` file.

## How to Run

1. **Clone or navigate to the project directory:**
   ```bash
   cd c:\Users\Ruhama\Desktop\IR\oromo_app
   ```

2. **Install the dependencies:**
   It is recommended to use a virtual environment, but you can install the dependencies globally as well:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application:**
   Run the `app.py` file using Python:
   ```bash
   python app.py
   ```

4. **Access the Web Interface:**
   Once the server is running, open your web browser and navigate to:
   ```
   http://localhost:5050
   ```

## Usage Guide

- **Stemmer Tab**: Enter a single word or a batch of sentences to see how the rule-based algorithm breaks down the words into their root stems.
- **Search Tab**: Enter search terms (e.g., "barumsa") to search across all indexed documents. The engine will intelligently match related forms (e.g., "barnoota", "barattootni").
- **Dictionary Tab**: Look up words in the COED dictionary. If an inflected word isn't found, use the "Stem & Look Up" feature to find the base word.
- **Corpus Tab**: 
  - Browse the collection of currently indexed documents.
  - Manually type and add a new text document.
  - **Upload** a `.txt`, `.pdf`, or `.docx` file from your local machine. The system will extract the text, add it to the corpus, and re-index the search engine to include your new document.

## Architecture Highlights
- The stemmer is based on the algorithm by *Debela Tesfaye & Ermias Abebe (2010)* in "Designing a Rule Based Stemmer for Afaan Oromo Text".
- Reduplication handling is fully implemented (e.g., converting plurals like *gaggabaabaa* to *gabaabaa*).
- The frontend is built with vanilla HTML/CSS/JS and is served dynamically by Flask, avoiding complex build steps while maintaining a modern, responsive UI.
