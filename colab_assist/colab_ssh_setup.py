# Google Colab SSH Setup Cell
# Run this in a new Colab notebook cell to establish SSH connection

import subprocess
import sys
import getpass
import time

print("🚀 Setting up SSH connection to Google Colab...")

# Step 1: Install required packages
print("\n📦 Installing required packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"], check=True)

# Step 2: Import packages
from pyngrok import ngrok
import os

# Step 3: Get ngrok authtoken
print("\n🔑 Setting up ngrok authentication...")
ngrok_token = getpass.getpass("Enter your ngrok authtoken: ")
ngrok.set_auth_token(ngrok_token)

# Step 4: Install and configure SSH server
print("\n🔧 Installing SSH server...")
subprocess.run(["apt-get", "update", "-q"], check=True)
subprocess.run(["apt-get", "install", "-q", "-y", "openssh-server"], check=True)

# Step 5: Configure SSH
print("\n⚙️ Configuring SSH server...")
subprocess.run(["service", "ssh", "start"], check=True)

# Set root password
root_password = "colab123"  # You can change this
subprocess.run(["bash", "-c", f"echo 'root:{root_password}' | chpasswd"], check=True)

# Enable root login
with open("/etc/ssh/sshd_config", "a") as f:
    f.write("\nPermitRootLogin yes\n")
    f.write("PasswordAuthentication yes\n")

# Restart SSH service
subprocess.run(["service", "ssh", "restart"], check=True)

# Step 6: Create ngrok tunnel
print("\n🌐 Creating ngrok tunnel...")
tunnel = ngrok.connect(22, "tcp")
tunnel_url = tunnel.public_url

# Parse connection details
import re
match = re.match(r'tcp://(.+):(\d+)', tunnel_url)
if match:
    host = match.group(1)
    port = match.group(2)
    
    print("\n" + "="*60)
    print("🎉 SSH CONNECTION READY!")
    print("="*60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Username: root")
    print(f"Password: {root_password}")
    print("\n📋 VS Code Connection String:")
    print(f"ssh root@{host} -p {port}")
    print("\n🔗 Or use this for VS Code Remote-SSH:")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print("="*60)
    
    # Keep tunnel alive
    print("\n⏳ Tunnel is active! Keep this cell running...")
    print("📝 Copy the connection details above and use them in VS Code")
    print("🛑 Stop this cell to close the tunnel")
    
    try:
        while True:
            time.sleep(30)
            print(".", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n🔌 Tunnel closed!")
        ngrok.disconnect(tunnel_url)
        ngrok.kill()
else:
    print("❌ Failed to parse tunnel URL")