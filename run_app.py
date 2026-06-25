"""
Smart Road Lane Line Detection System — Application Entry Point
================================================================
Run this file to start the Flask web server.
"""

from app.routes import app
import config

if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
