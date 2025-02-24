from flask import Flask, request, jsonify
from flask_cors import CORS

from qa_workflow.main import process

app = Flask(__name__)
CORS(app)  # Allow frontend requests

data = []

@app.route("/ingest", methods=["POST"])
def ingest_content():
    data = request.json
    urls = data.get("urls", [])

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    # for url in urls:
    #     try:
    #         response = requests.get(url)
    #         soup = BeautifulSoup(response.text, "html.parser")
    #         text = soup.get_text(separator=' ', strip=True)
    #         page_contents[url] = text
    #     except Exception as e:
    #         return jsonify({"error": f"Failed to fetch {url}: {str(e)}"}), 500
    #
    # return jsonify({"message": "Content ingested successfully", "urls": urls})

    for url in urls:
        data.append(url)

    return urls


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400


    # Simple search-based response generation
    response = process()

    return jsonify({"answer": response})


if __name__ == "__main__":
    app.run(debug=True)