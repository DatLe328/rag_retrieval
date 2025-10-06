from flask import Flask
from .routes import bp as rag_bp


def create_app():
    app = Flask(__name__)
    app.register_blueprint(rag_bp)
    return app
