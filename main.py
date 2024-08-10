from flask import Flask, request, jsonify, abort
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load models and data
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]: 
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url

def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1, -1), n_neighbors=6)

    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            books_list.append(j)
    
    # Ensure all data is serializable
    return {
        "books": books_list,
        "posters": poster_url
    }

@app.route('/recommend/', methods=['POST'])
def recommend():
    if not request.json or 'book_name' not in request.json:
        abort(400, description="Invalid request")
    
    book_name = request.json['book_name']
    
    if book_name not in book_names:
        abort(404, description="Book not found")
    
    recommendations = recommend_book(book_name)
    return jsonify(recommendations)

@app.route('/book_names/', methods=['GET'])
def get_book_names():
    return jsonify(book_names)

if __name__ == '__main__':
    app.run(debug=True)
