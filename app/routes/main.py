"""
Main routes for ECGWeb2 V2
"""
import os
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app.extensions import db
from app.models import Analysis, Result
from app.ml.model_handler import ECGAnalyzer

bp = Blueprint('main', __name__)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@bp.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Upload and analyze ECG file"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(request.url)

        file = request.files['file']

        # Check if filename is empty
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)

        # Validate file
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a CSV file.', 'danger')
            return redirect(request.url)

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            # Create analysis record
            analysis = Analysis(
                user_id=current_user.id,
                filename=filename,
                total_beats=0,
                status='processing'
            )
            db.session.add(analysis)
            db.session.commit()

            # Analyze the file
            analyzer = ECGAnalyzer()
            results = analyzer.analyze_file(filepath, analysis.id)

            # Update analysis record
            analysis.total_beats = results['total_beats']
            analysis.normal_count = results['normal_count']
            analysis.abnormal_count = results['abnormal_count']
            analysis.status = 'completed'

            # Save individual results
            for i in range(results['total_beats']):
                result = Result(
                    analysis_id=analysis.id,
                    beat_index=i,
                    prediction=results['predictions'][i],
                    confidence=results['confidences'][i],
                    graph_path=results['graph_paths'][i]
                )
                db.session.add(result)

            db.session.commit()

            # Clean up uploaded file
            os.remove(filepath)

            flash('Analysis completed successfully!', 'success')
            return redirect(url_for('main.results', analysis_id=analysis.id))

        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error processing upload: {str(e)}")
            flash(f'Error processing file: {str(e)}', 'danger')
            return redirect(request.url)

    return render_template('upload.html')


@bp.route('/results/<int:analysis_id>')
@login_required
def results(analysis_id):
    """Display analysis results"""
    analysis = Analysis.query.get_or_404(analysis_id)

    # Check if user owns this analysis
    if analysis.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('main.index'))

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = current_app.config['RESULTS_PER_PAGE']

    results = Result.query.filter_by(analysis_id=analysis_id).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return render_template('results.html', analysis=analysis, results=results)


@bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with analysis history"""
    page = request.args.get('page', 1, type=int)
    per_page = 10

    analyses = Analysis.query.filter_by(user_id=current_user.id)\
        .order_by(Analysis.upload_date.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)

    # Calculate statistics
    total_analyses = Analysis.query.filter_by(user_id=current_user.id).count()
    total_beats = db.session.query(db.func.sum(Analysis.total_beats))\
        .filter_by(user_id=current_user.id).scalar() or 0
    total_normal = db.session.query(db.func.sum(Analysis.normal_count))\
        .filter_by(user_id=current_user.id).scalar() or 0
    total_abnormal = db.session.query(db.func.sum(Analysis.abnormal_count))\
        .filter_by(user_id=current_user.id).scalar() or 0

    stats = {
        'total_analyses': total_analyses,
        'total_beats': total_beats,
        'total_normal': total_normal,
        'total_abnormal': total_abnormal
    }

    return render_template('dashboard.html', analyses=analyses, stats=stats)


@bp.route('/download/<int:analysis_id>')
@login_required
def download(analysis_id):
    """Download results as CSV"""
    analysis = Analysis.query.get_or_404(analysis_id)

    # Check if user owns this analysis
    if analysis.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('main.index'))

    # Create CSV file
    import pandas as pd
    from io import StringIO

    results = Result.query.filter_by(analysis_id=analysis_id).all()

    data = {
        'beat_id': [r.beat_index for r in results],
        'prediction': [r.prediction for r in results],
        'confidence': [r.confidence for r in results]
    }

    df = pd.DataFrame(data)
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    # Save to temporary file
    temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f'results_{analysis_id}.csv')
    with open(temp_path, 'w') as f:
        f.write(output.getvalue())

    return send_file(
        temp_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'ecg_results_{analysis.filename}'
    )


@bp.route('/delete/<int:analysis_id>', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    """Delete an analysis"""
    analysis = Analysis.query.get_or_404(analysis_id)

    # Check if user owns this analysis
    if analysis.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))

    try:
        # Clean up graph files
        analyzer = ECGAnalyzer()
        analyzer.cleanup_graphs(analysis_id)

        # Delete from database (cascade will delete results)
        db.session.delete(analysis)
        db.session.commit()

        flash('Analysis deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error deleting analysis: {str(e)}")
        flash('Error deleting analysis.', 'danger')

    return redirect(url_for('main.dashboard'))
