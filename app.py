
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import modelHandling
import modelHandling1

app = Flask(__name__)


@app.route('/')
def opening_Page():
    return render_template('/upload1.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        #output, pictures_name = modelHandling.give_prediction(f.filename, 'ECGmodel.h5')
        #print(pictures_name)
        output, all_data = modelHandling1.give_prediction(f.filename, 'ECGmodel.h5')

        # return render_template('/download.html',results=pictures_name,tables=[output.to_html(classes='data')], titles=output.columns.values)
        return render_template('/download1.html', allData=all_data)


@app.route('/download')
def downloader():
    filename='output.csv'
    return send_file(secure_filename(filename))


if __name__ == '__main__':
    app.run()
