"""
API routes for ECGWeb2 V2
"""
import os
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.ml.model_handler import ECGAnalyzer

bp = Blueprint('api', __name__)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0',
        'service': 'ECGWeb2'
    }), 200


@bp.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze ECG data via API

    Expects:
        - file: CSV file with ECG data

    Returns:
        JSON with analysis results
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Only CSV files are allowed.'
            }), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"api_{timestamp}_{filename}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Analyze the file
        analyzer = ECGAnalyzer()
        results = analyzer.analyze_file(filepath, analysis_id='api')

        # Clean up uploaded file
        os.remove(filepath)

        # Prepare response
        response_data = {
            'success': True,
            'summary': {
                'total_beats': results['total_beats'],
                'normal_count': results['normal_count'],
                'abnormal_count': results['abnormal_count'],
                'normal_percentage': round((results['normal_count'] / results['total_beats']) * 100, 2) if results['total_beats'] > 0 else 0,
                'abnormal_percentage': round((results['abnormal_count'] / results['total_beats']) * 100, 2) if results['total_beats'] > 0 else 0
            },
            'results': [
                {
                    'beat_id': i,
                    'prediction': results['predictions'][i],
                    'confidence': results['confidences'][i]
                }
                for i in range(results['total_beats'])
            ]
        }

        return jsonify(response_data), 200

    except Exception as e:
        current_app.logger.error(f"API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Batch analyze multiple ECG files

    Expects:
        - files: Multiple CSV files

    Returns:
        JSON with batch analysis results
    """
    try:
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400

        files = request.files.getlist('files')

        if not files:
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400

        results_list = []
        analyzer = ECGAnalyzer()

        for file in files:
            if file.filename == '' or not allowed_file(file.filename):
                continue

            # Save and analyze
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"batch_{timestamp}_{filename}"
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            try:
                file_results = analyzer.analyze_file(filepath, analysis_id='batch')

                results_list.append({
                    'filename': filename,
                    'success': True,
                    'summary': {
                        'total_beats': file_results['total_beats'],
                        'normal_count': file_results['normal_count'],
                        'abnormal_count': file_results['abnormal_count']
                    }
                })
            except Exception as e:
                results_list.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e)
                })
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

        return jsonify({
            'success': True,
            'total_files': len(files),
            'processed': len(results_list),
            'results': results_list
        }), 200

    except Exception as e:
        current_app.logger.error(f"Batch API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 50MB.'
    }), 413


@bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
