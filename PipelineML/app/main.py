from flask import Flask, request, render_template
from app.utils import fetch_clean_text_with_links
from app.predict import extract_products

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/extract", methods=["POST"])
def extract():
    try:
        url = request.form["url"]
        text = fetch_clean_text_with_links(url)
        products = extract_products(text)
        return render_template("results.html", products=products, url=url)
    except Exception as e:
        return render_template("results.html", products=[], url="", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
