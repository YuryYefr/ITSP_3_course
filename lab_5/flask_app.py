from flask import Flask, request, jsonify, render_template
import re

from connector import proc_selenium

app = Flask(__name__)


@app.route('/')
def index():
    values = proc_selenium()
    return render_template('index.html', values=values)


@app.route('/extract-name', methods=['POST'])
def extract_name():
    data = request.json
    url = data.get('url')

    # Use regex to extract the name from the URL
    match = re.search(r'([^-]+)-([^-]+)$', url)
    if match:
        name = f"{match.group(1)} {match.group(2)}"
        return jsonify({"name": name})
    else:
        return jsonify({"error": "Name not found in URL"}), 400


if __name__ == '__main__':
    app.run(debug=True)
