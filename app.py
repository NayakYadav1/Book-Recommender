from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import pypdf
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load existing recommendation data
popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))

app = Flask(__name__)

# Folder to save uploaded PDFs
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dictionary to store uploaded book summaries
book_summaries = {}


# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = pypdf.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


# Function to summarize text
def summarize_text(text, num_sentences=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join([str(sentence) for sentence in summary])


# TF-IDF Vectorizer for content similarity
vectorizer = TfidfVectorizer(stop_words="english")


# Recommend books based on uploaded PDF content
def recommend_similar_books(uploaded_summary):
    # Load book titles and authors from books.pkl
    book_titles = books['Book-Title'].values
    book_authors = books['Book-Author'].values
    book_summaries_list = []

    # Generate summaries for existing books (if not available)
    for title, author in zip(book_titles, book_authors):
        summary = f"This book, '{title}', is written by {author} and is a well-known book."
        book_summaries_list.append(summary)

    # Append uploaded book summary to compare
    book_titles_list = list(book_titles)
    book_summaries_list.append(uploaded_summary)

    # Convert text data to numerical format using TF-IDF
    book_vectors = vectorizer.fit_transform(book_summaries_list)
    uploaded_vector = book_vectors[-1]  # The last entry is the uploaded book

    # Compute similarity scores
    similarity = (book_vectors * uploaded_vector.T).toarray().flatten()

    # Get top 3 similar books (excluding the uploaded one)
    similar_indices = similarity.argsort()[-4:-1][::-1]

    # Return recommended book titles
    recommended_books = [book_titles_list[i] for i in similar_indices]
    return recommended_books


@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_rating'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')

    if user_input not in pt.index:
        return render_template('recommend.html', error="Book not found. Try another title.")

    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)

    return render_template('recommend.html', data=data)


# Handle PDF Upload and Recommend Books
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Extract and summarize book content
    text = extract_text_from_pdf(file_path)
    summary = summarize_text(text)

    # Store summary
    book_summaries[file.filename] = summary

    # Get recommended book titles
    recommended_titles = recommend_similar_books(summary)

    # Fetch book details (title, author, image) from `books.pkl`
    recommended_books_data = []
    for title in recommended_titles:
        book_info = books[books['Book-Title'] == title].drop_duplicates('Book-Title')
        if not book_info.empty:
            book_details = {
                "title": book_info['Book-Title'].values[0],
                "author": book_info['Book-Author'].values[0],
                "image": book_info['Image-URL-M'].values[0]
            }
            recommended_books_data.append(book_details)

    return jsonify({"uploaded_book": file.filename, "recommended_books": recommended_books_data})

if __name__ == '__main__':
    app.run(debug=True)
