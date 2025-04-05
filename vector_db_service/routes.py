from flask import Flask

app_name = 'vector-service'
app = Flask(app_name)

@app.route('/')
def home():
    return "Hello, Flask!"

if __name__ == '__main__':
    app.run(debug=True)