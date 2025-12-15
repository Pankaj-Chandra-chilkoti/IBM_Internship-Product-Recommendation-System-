# services.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendations(data_df, item_name, top_n=10,
                                  product_name_col='ProductName', tags_col='Category_SubCategory',
                                  image_url_col='ImageUrl', brand_col='Brand', price_col=None):
    if item_name not in data_df[product_name_col].values:
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_df[tags_col].fillna(''))
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    item_index = data_df[data_df[product_name_col] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    indices = [i[0] for i in top_similar_items]

    cols = [product_name_col, brand_col, image_url_col]
    if price_col:
        cols.append(price_col)

    result = data_df.iloc[indices][cols].rename(columns={
        product_name_col: 'ProductName',
        brand_col: 'Brand',
        image_url_col: 'ImageUrl',
        price_col: 'Price' if price_col else price_col
    })

    return result
