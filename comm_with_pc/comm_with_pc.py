#!/usr/bin/env python3
import socket
import send_commands_arduino
def start_server(host='0.0.0.0', port=12345):
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to the host and port
    server_socket.bind((host, port))
    print(f"Server started. Listening on {host}:{port}...")
    
    # Listen for incoming connections (increase backlog as needed)
    server_socket.listen(5)
    
    try:
        # Outer loop: keep accepting new connections
        while True:
            print("Waiting for a new connection...")
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
            
            try:
                # Inner loop: receive messages from the connected client
                while True:
                    data = conn.recv(1024)
                    if not data:
                        # No more data from client; connection is closed
                        break
                    # Decode and print the received data
                    message = data.decode().strip()
                    motorSide, direction, speedPercent = message.split(',')
                    send_commands_arduino.send_motor_command(motorSide,direction,speedPercent)
                    print("Received:", message)
                    
                    # Optionally, send a response back (echo the data)
                    conn.sendall(data)
            except Exception as e:
                print("Connection error:", e)
            finally:
                conn.close()
                print(f"Connection with {addr} closed.\n")
    except KeyboardInterrupt:
        print("Server interrupted by user.")
    finally:
        server_socket.close()
        print("Server shut down.")

if __name__ == "__main__":
    start_server()
