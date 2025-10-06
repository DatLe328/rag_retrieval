from flask import Flask
from .routes import bp as rag_bp
from dotenv import load_dotenv

load_dotenv()
def create_app():
    app = Flask(__name__)
    app.register_blueprint(rag_bp)
    return app
