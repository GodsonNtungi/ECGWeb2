"""
Configuration settings for ECGWeb2 V2
"""
import os
from datetime import timedelta

# Base directory
basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Base configuration"""

    # App settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'ecgweb.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload settings
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    GRAPHS_FOLDER = os.path.join(basedir, 'app', 'static', 'graphs')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    ALLOWED_EXTENSIONS = {'csv'}

    # Model settings
    MODEL_PATH = os.path.join(basedir, 'Models', 'ECGModelsmall.pkl')

    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)

    # Pagination
    RESULTS_PER_PAGE = 20

    @staticmethod
    def init_app(app):
        """Initialize application"""
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.GRAPHS_FOLDER, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

    # Use PostgreSQL in production
    # SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
