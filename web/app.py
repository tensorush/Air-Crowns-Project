from flask import Flask, render_template, request, send_file, \
                  send_from_directory, safe_join, abort

import ss_multiclass_unet

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():
  return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload():
  # File from client is in files['file']
  app.logger.info(request.files['file'])
  return send_file('../trees.png', as_attachment=True, mimetype='image/png')
  
if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0", debug=True)
