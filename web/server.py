import http.server
import socketserver
import webbrowser
import os

PORT = 5000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class SignVoxHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), SignVoxHandler) as httpd:
        url = f"http://localhost:{PORT}"
        print("=" * 60)
        print("🤟 SignVox Web Server is ACTIVE!")
        print(f"🌐 Address: {url}")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 60)
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server shut down.")