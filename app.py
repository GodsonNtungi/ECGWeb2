
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import modelHandling

app = Flask(__name__)


@app.route('/')
def opening_Page():
    return render_template('/upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        output = modelHandling.give_prediction(f.filename, 'ECGmodel.h5')
        return render_template('/download.html',tables=[output.to_html(classes='data')], titles=output.columns.values)


@app.route('/download')
def downloader():
    filename='output.csv'
    return send_file(secure_filename(filename))


if __name__ == '__main__':
    app.run()
