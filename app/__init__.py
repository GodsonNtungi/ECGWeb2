"""
Application factory for ECGWeb2 V2
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from config import config
from app.extensions import db, login_manager


def create_app(config_name='default'):
    """
    Application factory pattern

    Args:
        config_name: Configuration to use (development, production, testing)

    Returns:
        Flask app instance
    """
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)

    # Login manager settings
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    # Import models to register with SQLAlchemy
    from app.models import User, Analysis, Result

    # User loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register blueprints
    from app.routes import main, api, auth

    app.register_blueprint(main)
    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(auth, url_prefix='/auth')

    # Create database tables
    with app.app_context():
        db.create_all()

    # Set up logging
    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')

        file_handler = RotatingFileHandler(
            'logs/ecgweb.log',
            maxBytes=10240000,
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info('ECGWeb2 V2 startup')

    return app
