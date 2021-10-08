from flask import Flask, render_template, request, send_file, \
                  send_from_directory, safe_join, abort

from PIL import Image
import io
import ss_multiclass_unet

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():
  return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload():
  # File from client is in files['file']
  img_origin = Image.open(io.BytesIO(request.files['file'].read()))
  app.logger.info(img_origin)

  img_origin_path = 'rgb_origin.jpg'
  img_origin.save(img_origin_path)

  img_result_path = ss_multiclass_unet.predict_images(img_origin_path) 

  return send_file(f'{img_result_path}', as_attachment=True, mimetype='image/png')
  
if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0", debug=True)
