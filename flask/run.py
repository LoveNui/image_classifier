import argparse

from flask import Flask
from tensorflow.keras.models import load_model


app = Flask(__name__)

parser = argparse.ArgumentParser(description='Flask app')
parser.add_argument('--path', type=str, required=True,
                    help='A required path to weights')

args = parser.parse_args()
model = load_model(args.path)

with app.app_context():
    import views

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=5000, debug=True)