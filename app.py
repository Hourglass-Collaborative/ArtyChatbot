from flask import Flask, render_template, request, jsonify
from chat import generate_response, find_intent_match

app = Flask(__name__)


@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")

    response = generate_response(text)
    message = {"answer": response}

    return jsonify(message)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
    #app.run(debug=False)
    #