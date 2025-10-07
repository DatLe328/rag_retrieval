import os
from application import create_app
from rag_retrieval.config.settings import Settings
from flask_cors import CORS

app = create_app()
CORS(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Settings.RAG_FLASK_PORT, debug=Settings.RAG_FLASK_DEBUG)
