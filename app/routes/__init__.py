"""
Routes package initialization
"""
from flask import Blueprint

# Import blueprints
from app.routes.main import bp as main
from app.routes.api import bp as api
from app.routes.auth import bp as auth

__all__ = ['main', 'api', 'auth']
