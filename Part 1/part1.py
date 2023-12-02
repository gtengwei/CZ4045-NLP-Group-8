import gensim.downloader

# Downlaod 'word2vec-google-news-300' embeddings
google_news_vectors = gensim.downloader.load('word2vec-google-news-300')

# Get the embedding for 'king'
king = google_news_vectors['king']
print(king.shape)