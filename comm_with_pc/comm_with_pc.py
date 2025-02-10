#!/usr/bin/env python3
import socket

def start_server(host='0.0.0.0', port=12345):
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to the host and port
    server_socket.bind((host, port))
    print(f"Server started. Listening on {host}:{port}...")
    
    # Listen for incoming connections (1 means maximum queued connections)
    server_socket.listen(1)
    
    # Accept a connection
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    
    # Continuously receive data from the client
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                # No more data from client; exit the loop
                break
            # Decode and print the received data
            print("Received:", data.decode())
            
            # Optionally, send a response back to the client (echo the data)
            conn.sendall(data)
    except KeyboardInterrupt:
        print("Server interrupted by user.")
    finally:
        # Clean up the connection
        conn.close()
        server_socket.close()
        print("Server shut down.")

if __name__ == "__main__":
    start_server()