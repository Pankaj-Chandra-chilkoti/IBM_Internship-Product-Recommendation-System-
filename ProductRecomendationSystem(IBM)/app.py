from flask import Flask, render_template, request, session, redirect, url_for, jsonify, flash
import pandas as pd
import random
import os
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load product data from CSV files
train_data_bigbasket = pd.read_csv("models/bigbasket_cleaned.csv")
train_data_clean = pd.read_csv("models/clean_data.csv")

# Assign train_data_clean to trending_products for homepage display
trending_products = train_data_bigbasket # Change made here

# Database configuration for Flask-SQLAlchemy
app.secret_key = "500124743"  # Secret key for session management, crucial for security.
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:500124743@localhost/ecommerce_db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable tracking modifications for performance.
db = SQLAlchemy(app) # Initialize the SQLAlchemy instance with the Flask app.

# Define SQLAlchemy model for the 'signup' table.
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    full_name = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    profile_image = db.Column(db.String(255), default='default_profile.png')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define SQLAlchemy model for the 'signin' table.
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Cart(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    product_name = db.Column(db.String(255))
    image_url = db.Column(db.Text)
    price = db.Column(db.Float)
    quantity = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Recommendations functions ============================================================================================
# Helper function to truncate product names for display in cards.
def truncate(text, length):
    """
    Truncates a given text (product name) to a specified maximum length.
    If the text exceeds the length, "..." is appended.
    """
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

def content_based_recommendations(data_df, item_name, top_n=10,
                                  product_name_col='ProductName', tags_col='Category_SubCategory',
                                  image_url_col='ImageUrl', brand_col='Brand', price_col=None):

    # Ensure all required columns exist in the provided DataFrame
    required_cols = [product_name_col, tags_col, image_url_col, brand_col]
    if price_col:
        required_cols.append(price_col)

    for col in required_cols:
        if col not in data_df.columns:
            print(f"Error: Required column '{col}' not found in the DataFrame.")
            return pd.DataFrame()

    # Check if the requested item exists in the dataset under the specified product name column.
    if item_name not in data_df[product_name_col].values:
        print(f"Item '{item_name}' not found in the training data for recommendations (in '{product_name_col}').")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer.
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Fill NaN values in the tags column with empty strings to prevent errors during vectorization.
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(data_df[tags_col].fillna(''))

    # Calculate the cosine similarity matrix.
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the row index of the specified item in the DataFrame.
    item_index = data_df[data_df[product_name_col] == item_name].index[0]

    # Get the similarity scores for the chosen item against all other items.
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort the similar items by their similarity score in descending order.
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself).
    top_similar_items = similar_items[1:top_n+1]

    # Extract the original DataFrame indices of these top similar items.
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Retrieve the full details of the recommended items from the original DataFrame.
    # Dynamically select the product name, brand, image URL, and price columns.
    cols_to_select = [product_name_col, brand_col, image_url_col]
    if price_col:
        cols_to_select.append(price_col)

    recommended_items_details = data_df.iloc[recommended_item_indices][cols_to_select]

    # Rename columns to a consistent format for the template
    rename_dict = {
        product_name_col: 'ProductName',
        brand_col: 'Brand',
        image_url_col: 'ImageUrl'
    }
    if price_col:
        rename_dict[price_col] = 'Price'

    recommended_items_details = recommended_items_details.rename(columns=rename_dict)

    return recommended_items_details


# Flask Routes ===============================================================================

# List of predefined static image URLs. These are used as placeholders for product images
# if the actual ImageUrl from the dataset is not available or for trending products on the homepage.
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]

@app.route('/')
def index():
    # Get how many products to show, default is 20
    num_trending_products = request.args.get('limit', default=20, type=int)

    display_image_urls = []
    display_prices = []

    for index, product in trending_products.head(num_trending_products).iterrows():
        image_url = product.get('ImageUrl')  # since using bigbasket dataset
        price = product.get('Price', 'N/A')

        if pd.isna(image_url) or not isinstance(image_url, str) or not image_url.strip():
            display_image_urls.append(random.choice(random_image_urls))
        else:
            display_image_urls.append(image_url)

        display_prices.append(price)

    trending_display_name_col = 'ProductName'
    trending_display_image_url_col = 'ImageUrl'
    trending_display_price_col = 'Price'

    return render_template(
        'index.html',
        trending_products=trending_products.head(num_trending_products),
        truncate=truncate,
        random_product_image_urls=display_image_urls,
        trending_product_prices=display_prices,
        trending_display_name_col=trending_display_name_col,
        trending_display_image_url_col=trending_display_image_url_col,
        trending_display_price_col=trending_display_price_col,
        current_limit=num_trending_products   
    )



@app.route('/main')
def main():

    content_based_rec = pd.DataFrame()  
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]

    return render_template('main.html',
                           content_based_rec=content_based_rec, 
                           truncate=truncate,
                           random_product_image_urls=random_product_image_urls)

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/logout')
def logout():

    session.clear() # Removes all session-related data (e.g., user ID, username).
    return redirect(url_for('index')) # Redirects to the root URL (homepage).


@app.route("/index")
def indexredirect():

    return redirect(url_for('index')) # Simply redirect to the main index route.

@app.route("/signup", methods=['POST','GET'])
def signup():

    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if a user with the same username or email already exists in the database.
        existing_user = Signup.query.filter((Signup.username == username) | (Signup.email == email)).first()

        # Prepare display_image_urls and prices for rendering index.html for signup/signin paths
        num_trending_products = 20
        display_image_urls = []
        display_prices = []
        for index, product in trending_products.head(num_trending_products).iterrows():
            image_url = product.get('ImageURL')
            price = product.get('Price', 'N/A')

            if pd.isna(image_url) or not isinstance(image_url, str) or not image_url.strip():
                display_image_urls.append(random.choice(random_image_urls))
            else:
                display_image_urls.append(image_url)
            display_prices.append(price)

        trending_display_name_col = 'Name'
        trending_display_image_url_col = 'ImageURL'
        trending_display_price_col = 'Price'

        if existing_user:
            return render_template('index.html', trending_products=trending_products.head(num_trending_products), truncate=truncate,
                            random_product_image_urls=display_image_urls, trending_product_prices=display_prices,
                            signup_message='User signed up unsuccessful! Username or Email already exists.',
                            trending_display_name_col=trending_display_name_col,
                            trending_display_image_url_col=trending_display_image_url_col,
                            trending_display_price_col=trending_display_price_col
                            )

        # Hash the password for security before storing it in the database.
        hashed_password = generate_password_hash(password)
        # Create a new Signup object.
        new_user = Signup(username=username, email=email, password=hashed_password)
        db.session.add(new_user) # Add the new user record to the database session.
        db.session.commit() # Commit the transaction to save the changes to the database.

        # On successful signup, render the homepage with a success message.
        return render_template('index.html', trending_products=trending_products.head(num_trending_products), truncate=truncate,
                               random_product_image_urls=display_image_urls, trending_product_prices=display_prices,
                               signup_message='User signed up successfully!',
                               trending_display_name_col=trending_display_name_col,
                               trending_display_image_url_col=trending_display_image_url_col,
                               trending_display_price_col=trending_display_price_col
                               )

@app.route('/signin', methods=['POST'])
def signin():

    username = request.form['signinUsername']
    password = request.form['signinPassword']

    # Query the database to find a user with the provided username.
    user = Signup.query.filter_by(username=username).first()

    # Prepare display_image_urls and prices for rendering index.html for signup/signin paths
    num_trending_products = 20
    display_image_urls = []
    display_prices = []
    for index, product in trending_products.head(num_trending_products).iterrows():
        image_url = product.get('ImageURL')
        price = product.get('Price', 'N/A')

        if pd.isna(image_url) or not isinstance(image_url, str) or not image_url.strip():
            display_image_urls.append(random.choice(random_image_urls))
        else:
            display_image_urls.append(image_url)
        display_prices.append(price)

    trending_display_name_col = 'Name'
    trending_display_image_url_col = 'ImageURL'
    trending_display_price_col = 'Price'

    # Check if user was found AND if the provided password matches the stored hashed password.
    if user and check_password_hash(user.password, password):
        session['user_id'] = user.id       # Store user ID in session for later use (e.g., authentication checks).
        session['username'] = user.username # Store username in session for personalized greetings.
        session['profile_image'] = user.profile_image

        # On successful signin, render the homepage with a success message.
        return render_template('index.html', trending_products=trending_products.head(num_trending_products), truncate=truncate,
                               random_product_image_urls=display_image_urls, trending_product_prices=display_prices,
                               signup_message='User signed in successfully!',
                               trending_display_name_col=trending_display_name_col,
                               trending_display_image_url_col=trending_display_image_url_col,
                               trending_display_price_col=trending_display_price_col
                               )
    else:
        # If authentication fails, render the homepage with an error message.
        return render_template('index.html', trending_products=trending_products.head(num_trending_products), truncate=truncate,
                               random_product_image_urls=display_image_urls, trending_product_prices=display_prices,
                               signup_message='Invalid username or password!',
                               trending_display_name_col=trending_display_name_col,
                               trending_display_image_url_col=trending_display_image_url_col,
                               trending_display_price_col=trending_display_price_col)


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():

    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr_str = request.form.get('nbr')
        dataset_choice = request.form.get('dataset_choice')

        # Safely convert to integer
        try:
            nbr = int(nbr_str) if nbr_str else 10
        except ValueError:
            nbr = 10

        price_col_name = None # Initialize price column name
        # Load correct dataset
        if dataset_choice == 'clean_data':
            current_train_data = train_data_clean
            product_name_col = 'Name'
            tags_col = 'Tags'
            image_url_col = 'ImageURL'
            brand_col = 'Brand'
            price_col_name = 'Price' # Set price column for clean_data
        elif dataset_choice == 'bigbasket_cleaned': # Added this block
            current_train_data = train_data_bigbasket
            product_name_col = 'ProductName'
            tags_col = 'Category_SubCategory'
            image_url_col = 'ImageUrl'
            brand_col = 'Brand'
            price_col_name = 'Price' # Assume 'Price' column exists in bigbasket_cleaned.csv
        else:
            current_train_data = train_data_bigbasket # Default to bigbasket if unknown
            product_name_col = 'ProductName'
            tags_col = 'Category_SubCategory'
            image_url_col = 'ImageUrl'
            brand_col = 'Brand'


        # Call recommendation function
        content_based_rec = content_based_recommendations(
            current_train_data, prod, top_n=nbr,
            product_name_col=product_name_col,
            tags_col=tags_col,
            image_url_col=image_url_col,
            brand_col=brand_col,
            price_col=price_col_name # Pass the price column name
        )

        if content_based_rec.empty:
            message = f"No recommendations available for '{prod}'. Please try another."
            return render_template('main.html',
                                   message=message,
                                   content_based_rec=pd.DataFrame(),
                                   truncate=truncate,
                                   random_product_image_urls=[random.choice(random_image_urls) for _ in range(len(trending_products))]
                                   )
        else:
            return render_template('main.html',
                                   content_based_rec=content_based_rec,
                                   truncate=truncate,
                                   random_product_image_urls=[random.choice(random_image_urls) for _ in range(len(content_based_rec))] # Use length of recommendations
                                   )

    else:
        return render_template('main.html',
                               content_based_rec=pd.DataFrame(),
                               truncate=truncate,
                               random_product_image_urls=[random.choice(random_image_urls) for _ in range(len(trending_products))]
                               )


# New route for product suggestions
@app.route('/suggest_products')
def suggest_products():
    query = request.args.get('query', '').lower()
    dataset = request.args.get('dataset', '')

    if dataset == "clean_data":
        df = pd.read_csv('models/clean_data.csv')
        product_col = 'Name'
    elif dataset == "bigbasket_cleaned":
        df = pd.read_csv('models/bigbasket_cleaned.csv')
        product_col = 'ProductName'
    else:
        return jsonify([])

    suggestions = df[product_col].dropna().unique()
    matches = [name for name in suggestions if query in name.lower()]
    return jsonify(matches[:10])  # limit to top 10

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    if 'username' not in session:
        return redirect(url_for('signin'))  # must be logged in

    product_name = request.form['product_name']
    image_url = request.form['image_url']
    price = float(request.form['price'])
    username = session['username']

    # Check if item already in cart for this user
    existing = Cart.query.filter_by(username=username, product_name=product_name).first()
    if existing:
        existing.quantity += 1
    else:
        new_item = Cart(username=username, product_name=product_name, image_url=image_url, price=price)
        db.session.add(new_item)

    db.session.commit()
    flash("✅ Product added to cart!")
    return redirect(url_for('view_cart'))

@app.route('/cart')
def view_cart():
    if 'username' not in session:
        return redirect(url_for('signin'))

    items = Cart.query.filter_by(username=session['username']).all()
    total = sum(item.price * item.quantity for item in items)
    return render_template('cart.html', cart_items=items, total=total)

@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    if 'username' not in session:
        return redirect(url_for('signin'))

    item_id = request.form['item_id']
    item = Cart.query.filter_by(id=item_id, username=session['username']).first()

    if item:
        db.session.delete(item)
        db.session.commit()

    return redirect(url_for('view_cart'))

@app.route('/update_quantity', methods=['POST'])
def update_quantity():
    if 'username' not in session:
        return redirect(url_for('signin'))

    cart_id = request.form['cart_id']
    quantity = request.form['quantity']

    item = Cart.query.filter_by(id=cart_id, username=session['username']).first()

    if item and quantity.isdigit():
        item.quantity = int(quantity)
        db.session.commit()

    return redirect(url_for('view_cart'))

@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('signin'))

    user = Signup.query.filter_by(username=session['username']).first()

    if not user:
        return redirect(url_for('logout'))  # fallback if user not found

    return render_template('profile.html', user=user)

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if 'username' not in session:
        return redirect(url_for('signin'))

    user = Signup.query.filter_by(username=session['username']).first()

    if request.method == 'POST':
        old_password = request.form['old_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        if not check_password_hash(user.password, old_password):
            return render_template('change_password.html', error="Incorrect current password.")

        if new_password != confirm_password:
            return render_template('change_password.html', error="New passwords do not match.")

        # Hash and update new password
        user.password = generate_password_hash(new_password)
        db.session.commit()
        return render_template('change_password.html', success="Password updated successfully!")

    return render_template('change_password.html')

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'username' not in session:
        return redirect(url_for('signin'))

    user = Signup.query.filter_by(username=session['username']).first()

    if request.method == 'POST':
        new_username = request.form['username']
        new_email = request.form['email']
        full_name = request.form['full_name']
        phone = request.form['phone']
        address = request.form['address']

        # Check duplicates
        existing_user = Signup.query.filter(Signup.username == new_username, Signup.id != user.id).first()
        existing_email = Signup.query.filter(Signup.email == new_email, Signup.id != user.id).first()

        if existing_user:
            return render_template('edit_profile.html', user=user, error="Username already taken.")
        if existing_email:
            return render_template('edit_profile.html', user=user, error="Email already registered.")

        user.username = new_username
        user.email = new_email
        user.full_name = full_name
        user.phone = phone
        user.address = address

        # Profile image handling
        if 'profile_image' in request.files:
            file = request.files['profile_image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                user.profile_image = filename

        db.session.commit()
        session['username'] = new_username
        session['profile_image'] = user.profile_image

        return redirect(url_for('profile'))

    return render_template('edit_profile.html', user=user)


if __name__ == '__main__':
    # Runs the Flask application.
    # 'debug=True' enables debug mode, which provides helpful error messages
    # and automatically reloads the server on code changes.
    app.run(debug=True)
