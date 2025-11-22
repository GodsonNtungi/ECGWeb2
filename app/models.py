"""
Database models for ECGWeb2 V2
"""
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app.extensions import db


class User(UserMixin, db.Model):
    """User model"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    # Relationships
    analyses = db.relationship('Analysis', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'


class Analysis(db.Model):
    """ECG Analysis model"""
    __tablename__ = 'analyses'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(256), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    total_beats = db.Column(db.Integer, nullable=False)
    normal_count = db.Column(db.Integer, default=0)
    abnormal_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='processing')  # processing, completed, failed

    # Relationships
    results = db.relationship('Result', backref='analysis', lazy='dynamic', cascade='all, delete-orphan')

    @property
    def normal_percentage(self):
        """Calculate percentage of normal beats"""
        if self.total_beats == 0:
            return 0
        return round((self.normal_count / self.total_beats) * 100, 2)

    @property
    def abnormal_percentage(self):
        """Calculate percentage of abnormal beats"""
        if self.total_beats == 0:
            return 0
        return round((self.abnormal_count / self.total_beats) * 100, 2)

    def __repr__(self):
        return f'<Analysis {self.id}: {self.filename}>'


class Result(db.Model):
    """Individual beat result model"""
    __tablename__ = 'results'

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id'), nullable=False, index=True)
    beat_index = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(20), nullable=False)  # Normal or Abnormal
    confidence = db.Column(db.Float, nullable=False)
    graph_path = db.Column(db.String(256))

    def __repr__(self):
        return f'<Result {self.id}: {self.prediction}>'
