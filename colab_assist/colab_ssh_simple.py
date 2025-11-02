# Simple SSH Setup for Google Colab
# Copy and paste this into a Colab cell

# Install pyngrok
!pip install -q pyngrok

# Setup SSH
!apt-get update -q
!apt-get install -q -y openssh-server

# Start SSH service
!service ssh start

# Set password for root user
!echo 'root:colab123' | chpasswd

# Configure SSH to allow root login
!echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
!echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
!service ssh restart

# Create ngrok tunnel
from pyngrok import ngrok
import getpass

# Enter your ngrok authtoken when prompted
ngrok_token = getpass.getpass("Enter your ngrok authtoken: ")
ngrok.set_auth_token(ngrok_token)

# Create tunnel
tunnel = ngrok.connect(22, "tcp")
print(f"SSH Tunnel URL: {tunnel.public_url}")

# Parse connection details
import re
match = re.match(r'tcp://(.+):(\d+)', tunnel.public_url)
if match:
    host = match.group(1)
    port = match.group(2)
    print(f"\n🎉 CONNECTION READY!")
    print(f"SSH Command: ssh root@{host} -p {port}")
    print(f"Password: colab123")
    print(f"\nFor VS Code Remote-SSH:")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"User: root")
    
# Keep tunnel alive (run this in a separate cell if needed)
import time
try:
    print("\n⏳ Keeping tunnel alive... (Ctrl+C to stop)")
    while True:
        time.sleep(30)
        print(".", end="")
except KeyboardInterrupt:
    print("\n🔌 Tunnel closed!")
    ngrok.disconnect(tunnel.public_url)
    ngrok.kill()