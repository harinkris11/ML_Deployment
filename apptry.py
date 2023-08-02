from flask import Flask
app = Flask(__name__)

@app.route('/')
def Welcome():
    return "welcome to my project"




if __name__ =='__main__':
    app.run()