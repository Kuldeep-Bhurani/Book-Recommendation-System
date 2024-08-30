from flask import Flask, render_template, request, send_from_directory
import pickle
import numpy as np
import os

popularity_df = pickle.load(open('./models/popularity.pkl', 'rb'))
books_pt = pickle.load(open('./models/books_pt.pkl', 'rb'))
books_n_popularity_df = pickle.load(
    open('./models/books_n_popularity_df.pkl', 'rb'))
model_knn = pickle.load(open('./models/model_knn.pkl', 'rb'))

app = Flask(__name__)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/')
def index():
    return render_template('index.html', book_name=list(popularity_df['Book-Title'].values), author=list(popularity_df['Book-Author'].values), img_url=list(popularity_df['Image-URL-L'].values), year_of_publication=list(popularity_df['Year-Of-Publication'].values), rating=list(popularity_df['avg_ratings'].values), votes=list(popularity_df['num_ratings'].values))


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html', book_name=list(popularity_df['Book-Title'].values))


@app.route('/recommend_books', methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    if user_input in books_n_popularity_df.values:
        query_index = np.where(books_pt.index == user_input)[0][0]
        distances, indices = model_knn.kneighbors(
            books_pt.iloc[query_index, :].values.reshape(1, -1), n_neighbors=11)
        data = []
        for i in range(1, len(distances.flatten())):
            item = []
            temp_df = books_n_popularity_df[books_n_popularity_df['Book-Title']
                                            == books_pt.index[indices.flatten()[i]]]
            temp_df.drop_duplicates('Book-Title')
            item.extend(list(temp_df['Book-Title'].values))
            item.extend(list(temp_df['Book-Author'].values))
            item.extend(list(temp_df['Image-URL-L'].values))
            item.extend(list(temp_df['Year-Of-Publication'].values))
            item.extend(list(temp_df['avg_ratings'].values))
            item.extend(list(temp_df['num_ratings'].values))
            data.append(item)
        return render_template('recommend.html', data=data, book_name=list(popularity_df['Book-Title'].values))
    else:
        return render_template('recommend.html', err='This book was not found in the database, try another book!', book_name=list(popularity_df['Book-Title'].values))


if __name__ == '__main__':
    app.run()
