import time

from flask import Flask, request
from flask_cors import CORS

# Comment for dev
# from qna_service.openai_client import request_gpt_no_rag
# from qna_service.openai_client import run_rag_pipeline

# Uncomment for dev
from qna_service.openai_client import request_gpt_no_rag
from qna_service.openai_client import run_rag_pipeline

app = Flask(__name__)
CORS(app)


@app.route("/v1/chat/simple", methods=["POST"])
def process_message_simple_test():
    # Sanity test of the API functioning. Use for debugging purposes
    data = request.json
    messages = data.get("messages", [])
    uppercased_message = messages[-1]["content"].upper()
    time.sleep(2)
    # Returns capitalized user text as response
    return "MediMate says: " + uppercased_message


@app.route("/v1/chat/openai", methods=["POST"])
def process_message_with_chatgpt():
    data = request.json
    messages = data.get("messages", [])
    medical_history = data.get("medicalHistory", "")
    model = request.args.get("model", default="gpt-3.5-turbo-instruct")
    response = request_gpt_no_rag(messages, medical_history, model)
    return response


@app.route("/v1/chat/openai/rag", methods=["POST"])
def process_message_with_rag_chatgpt():
    data = request.json
    messages = data.get("messages", [])
    medical_history = data.get("medicalHistory", "")
    model = request.args.get("model", default="gpt-3.5-turbo-instruct")
    dataset = request.args.get("dataset", default="nfcorpus")
    response = run_rag_pipeline(messages, medical_history, model, dataset)
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
