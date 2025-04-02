import os
import logging

from flask import Flask, request, jsonify
from dotenv import load_dotenv

from .utils import process_music
from .chat import chat_with_llm

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/playlist', methods=['POST'])
def create_playlist():
    """
    Receives a playlist request as a prompt and returns the LLM's response (list of songs).
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        music_files = data.get('music_files')  # Expecting a list of file paths or file-like objects

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        if not music_files:
            return jsonify({"error": "Music files are required"}), 400

        # Process music files (assuming process_music can handle a list of files)
        retriever = process_music(music_files)

        # Invoke chat with LLM
        rag_chain = chat_with_llm(retriever)
        response = rag_chain.invoke(prompt)

        logging.info(f"LLM Response: {response}")
        return jsonify({"playlist": response}), 200

    except Exception as e:
        logging.exception("Error processing request")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)