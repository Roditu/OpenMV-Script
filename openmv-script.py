import sensor, image, time, os, tf, uos, gc, network, socket, json, machine
from pyb import Pin, Timer

def create_ap_mode():
    nic = network.WINC(mode=network.WINC.MODE_AP)
    nic.start_ap(ssid='OpenMV_AP', key='1234567890', security=network.WINC.WPA_PSK, channel=2)
    print("Access Point Created. SSID: OpenMV_AP, Password: 1234567890")
    return nic

def start_web_server():
    addr = socket.getaddrinfo('0.0.0.0', 80)[0][-1]
    s = socket.socket()
    s.bind(addr)
    s.listen(5)
    print('Listening on', addr)

    while True:
        cl, addr = s.accept()
        print('Client connected from', addr)
        request = cl.recv(1024)
        request = str(request)
        print('Content = %s' % request)

        if 'ssid' in request and 'password' in request:
            ssid_start = request.find('ssid=') + len('ssid=')
            ssid_end = request.find('&', ssid_start)
            ssid = request[ssid_start:ssid_end]

            password_start = request.find('password=') + len('password=')
            password_end = request.find(' ', password_start)
            password = request[password_start:password_end]

            with open('config.json', 'w') as f:
                f.write(json.dumps({"ssid": ssid, "password": password}))

            response = 'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n' \
                       '<html><body><h1>Configuration Saved!</h1>' \
                       '<p>SSID: {}</p><p>Password: {}</p>' \
                       '<p>Please restart the device.</p></body></html>'.format(ssid, password)
            cl.send(response)
            cl.close()
            machine.reset()
            return

        response = 'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n' \
                   '<html><body><h1>Enter WiFi Credentials</h1>' \
                   '<form action="/" method="GET">' \
                   'SSID: <input type="text" name="ssid"><br>' \
                   'Password: <input type="text" name="password"><br>' \
                   '<input type="submit" value="Submit">' \
                   '</form></body></html>'
        cl.send(response)
        cl.close()

def connect_to_wifi():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        ssid = config.get('ssid')
        password = config.get('password')
        if not ssid or not password:
            raise Exception('SSID or password not found in config.json')
    except Exception as e:
        print('Failed to load WiFi credentials, starting AP mode.')
        create_ap_mode()
        start_web_server()
        return None

    nic = network.WINC()
    print("Connecting to WiFi...")
    nic.connect(ssid, key=password)

    timeout = 15
    start_time = time.time()
    while not nic.isconnected():
        if time.time() - start_time > timeout:
            print("Failed to connect to WiFi, starting AP mode.")
            create_ap_mode()
            start_web_server()
            return None
        time.sleep(1)
        print("...")

    print("Connected to WiFi!")
    return nic

def setup_buzzer():
    p = Pin('P9')
    tim = Timer(4, freq=1000)
    ch = tim.channel(3, Timer.PWM, pin=p)
    ch.pulse_width_percent(100)
    return ch

def play_buzzer(ch):
    ch.pulse_width_percent(50)

def stop_buzzer(ch):
    ch.pulse_width_percent(100)

buzzer_channel = setup_buzzer()

nic = connect_to_wifi()
if not nic:
    print("No WiFi connection established.")
    while True:
        time.sleep(1)

ip_config = nic.ifconfig()
ip_address = ip_config[0]
print("IP Address:", ip_address)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (ip_address, 1024)
sock.bind(server_address)
sock.listen(1)
print(f"Server listening on {server_address}")

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((240, 240))
sensor.skip_frames(time=2000)

net = None
labels = None

try:
    net = tf.load("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    print(e)
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

clock = time.clock()

while True:
    print("Waiting for a connection...")
    connection, client_address = sock.accept()
    connection.setblocking(False)

    try:
        print(f"Connection from {client_address}")

        buffer = ""

        while True:

            try:
                data = connection.recv(1024).decode()
                if data:
                    buffer += data

                    while "\n" in buffer:
                        print("Trying to read ...")
                        line, buffer = buffer.split("\n", 1)
                        try:
                            json_data = json.loads(line)
                            print(f"JSON data: {json_data}")
                            if json_data.get("drowsinessStatus") in ["Unhealthy", "MicroSleep"]:
                                play_buzzer(buzzer_channel)
                            else:
                                stop_buzzer(buzzer_channel)
                        except ValueError as e:
                            print("Failed to decode JSON:", e)
                else:
                    print("No data received in this cycle. Continuing to wait for data...")
            except OSError as e:

                if e.args[0] != 11:
                    print(f"Socket error: {e}")
                    break


            try:
                clock.tick()
                img = sensor.snapshot()

                for obj in net.classify(img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
                    print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
                    img.draw_rectangle(obj.rect())
                    predictions_list = list(zip(labels, obj.output()))

                    for i in range(len(predictions_list)):
                        print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))

                    best_prediction = max(predictions_list, key=lambda x: x[1])
                    best_label = best_prediction[0]
                    best_confidence = best_prediction[1]
                    print("Best Prediction: %s = %f" % (best_label, best_confidence))

                    data_to_send = json.dumps({"label": best_label, "confidence": best_confidence}) + "\n"
                    connection.sendall(data_to_send.encode())
                    print(f"Sent: {data_to_send}")

                print(clock.fps(), "fps")
            except Exception as e:
                print(f"Failed during classification or sending data: {e}")
                break

            time.sleep(0.1)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()
        print(f"Disconnected from {client_address}")
        print("Ready to accept a new connection")
