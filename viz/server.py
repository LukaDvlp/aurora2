from flask import Flask, render_template, request
import cv2
import socket

app = Flask(__name__)
sock = None

@app.route('/')
def show_main():
    return render_template('index.html')

@app.route('/send_goal', methods=['POST'])
def send_goal():
    #startUV = (float(request.form['startU']), float(request.form['startV']))
    goalUV  = (float(request.form['goalU']), float(request.form['goalV']))
    print goalUV
    sock.send('xy{:.2f},{:.2f}'.format(goalUV[0], goalUV[1]))
    return 'ok'
    
    

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #sock.connect(('192.168.201.10', 5557))
    app.run(debug=True)
    #app.run(debug=True, host='192.168.201.10')
