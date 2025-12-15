# services.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendations(data_df, item_name, top_n=10,
                                  product_name_col='ProductName', tags_col='Category_SubCategory',
                                  image_url_col='ImageUrl', brand_col='Brand', price_col=None):
    required_cols = [product_name_col, tags_col, image_url_col, brand_col]
    if price_col:
        required_cols.append(price_col)

    for col in required_cols:
        if col not in data_df.columns:
            return pd.DataFrame()

    if item_name not in data_df[product_name_col].values:
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(data_df[tags_col].fillna(''))
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    item_index = data_df[data_df[product_name_col] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]

    cols_to_select = [product_name_col, brand_col, image_url_col]
    if price_col:
        cols_to_select.append(price_col)

    recommended_items_details = data_df.iloc[recommended_item_indices][cols_to_select]

    rename_dict = {
        product_name_col: 'ProductName',
        brand_col: 'Brand',
        image_url_col: 'ImageUrl'
    }
    if price_col:
        rename_dict[price_col] = 'Price'

    recommended_items_details = recommended_items_details.rename(columns=rename_dict)

    return recommended_items_details
