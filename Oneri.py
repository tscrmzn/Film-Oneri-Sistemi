import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Veri Okuma 
ratings_path = "ratings.dat"
movies_path = "movies.dat"

df_ratings = pd.read_csv(ratings_path, sep='::', engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
df_ratings = df_ratings.drop('timestamp', axis=1)

df_movies = pd.read_csv(movies_path, sep='::', engine='python', names=['movieId', 'title', 'genres'], encoding='latin1')

# ID Encoding 
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df_ratings['user'] = user_encoder.fit_transform(df_ratings['userId'])
df_ratings['item'] = item_encoder.fit_transform(df_ratings['movieId'])

num_users = df_ratings['user'].nunique()
num_items = df_ratings['item'].nunique()

# Train-Test Split
X = df_ratings[['user', 'item']].values
y = df_ratings['rating'].values.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
def create_ncf_model(num_users, num_items, embedding_dim=32):
    user_input = Input(shape=(1,), name='user')
    item_input = Input(shape=(1,), name='item')

    user_embedding = Embedding(num_users, embedding_dim, name='user_emb')(user_input)
    item_embedding = Embedding(num_items, embedding_dim, name='item_emb')(item_input)

    user_vec = Flatten()(user_embedding)
    item_vec = Flatten()(item_embedding)

    concat = Concatenate()([user_vec, item_vec])
    x = Dense(128, activation='relu')(concat)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1)(x)

    model = Model(inputs=[user_input, item_input], outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

model = create_ncf_model(num_users, num_items, embedding_dim=32)

# Eğitim
model.fit([X_train[:, 0], X_train[:, 1]], y_train,
          epochs=10, batch_size=1024, validation_split=0.1, verbose=1)


loss = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test)
print(f"Test Loss: {loss:.4f}")

# Tahmin fonksiyonu
def predict_rating(user_id, movie_id):
    encoded_user = user_encoder.transform([user_id])[0]
    encoded_movie = item_encoder.transform([movie_id])[0]
    pred = model.predict([np.array([encoded_user]), np.array([encoded_movie])])[0][0]
    movie_info = df_movies[df_movies['movieId'] == movie_id].iloc[0]
    return f"Kullanıcı {user_id} için '{movie_info['title']}' filmi ({movie_info['genres']}) tahmini puanı: {pred:.2f}"

# Örnek test
print(predict_rating(555, 230))
