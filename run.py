#!/usr/bin/env python3
"""
ECGWeb2 V2 - Application Entry Point
Run this file to start the application
"""
import os
from app import create_app
from app.extensions import db
from app.models import User, Analysis, Result

# Get configuration from environment or default to development
config_name = os.getenv('FLASK_ENV', 'development')
app = create_app(config_name)


@app.shell_context_processor
def make_shell_context():
    """
    Make database models available in flask shell
    Usage: flask shell
    """
    return {
        'db': db,
        'User': User,
        'Analysis': Analysis,
        'Result': Result
    }


@app.cli.command()
def init_db():
    """
    Initialize the database
    Usage: flask init-db
    """
    db.create_all()
    print('✓ Database initialized successfully!')


@app.cli.command()
def create_admin():
    """
    Create an admin user
    Usage: flask create-admin
    """
    from getpass import getpass

    username = input('Enter username: ')
    email = input('Enter email: ')
    password = getpass('Enter password: ')
    confirm_password = getpass('Confirm password: ')

    if password != confirm_password:
        print('✗ Passwords do not match!')
        return

    # Check if user exists
    if User.query.filter_by(username=username).first():
        print('✗ Username already exists!')
        return

    if User.query.filter_by(email=email).first():
        print('✗ Email already registered!')
        return

    # Create user
    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    print(f'✓ Admin user "{username}" created successfully!')


@app.cli.command()
def test_model():
    """
    Test the ML model with sample data
    Usage: flask test-model
    """
    from app.ml.model_handler import ECGAnalyzer
    import os

    sample_file = 'TestData/smalldata.csv'

    if not os.path.exists(sample_file):
        print(f'✗ Sample file not found: {sample_file}')
        return

    print(f'Testing model with {sample_file}...')

    try:
        analyzer = ECGAnalyzer()
        results = analyzer.analyze_file(sample_file, analysis_id='test')

        print(f'\n✓ Analysis completed successfully!')
        print(f'  Total beats: {results["total_beats"]}')
        print(f'  Normal: {results["normal_count"]} ({results["normal_count"]/results["total_beats"]*100:.1f}%)')
        print(f'  Abnormal: {results["abnormal_count"]} ({results["abnormal_count"]/results["total_beats"]*100:.1f}%)')

    except Exception as e:
        print(f'✗ Error: {str(e)}')


if __name__ == '__main__':
    # Run the application
    # For development: python run.py
    # For production: gunicorn -w 4 -b 0.0.0.0:8000 run:app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=(config_name == 'development')
    )
