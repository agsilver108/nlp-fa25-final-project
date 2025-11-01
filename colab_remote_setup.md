# VS Code + Google Colab Pro Remote Setup

## Method 1: Direct Upload (Recommended for our project)
1. Open VS Code with your notebook
2. Go to [colab.research.google.com](https://colab.research.google.com)
3. File → Upload notebook → Select `NLP_Final_Project_Colab.ipynb`
4. Runtime → Change runtime type → GPU (T4/A100)
5. Run cells to execute training

## Method 2: GitHub Sync (Current Method - Working Great!)
```bash
# Your current optimized workflow:
# 1. Edit in VS Code
# 2. Commit changes: git add . && git commit -m "updates" && git push
# 3. In Colab: !git pull origin main
# 4. Execute training with GPU acceleration
```

## Method 3: VS Code Remote to Colab (Advanced)
### Setup SSH Tunnel to Colab:

1. **In Colab, run this setup cell:**
```python
# Install necessary packages
!pip install -q pyngrok
from pyngrok import ngrok
import getpass

# Set up ngrok authtoken (get from https://ngrok.com)
ngrok_token = getpass.getpass("Enter your ngrok authtoken: ")
ngrok.set_auth_token(ngrok_token)

# Start SSH server
!apt-get install -q -y openssh-server
!service ssh start
!echo 'root:password123' | chpasswd

# Create ngrok tunnel
tunnel = ngrok.connect(22, "tcp")
print(f"SSH connection: {tunnel.public_url}")
```

2. **In VS Code, install Remote-SSH extension:**
```vscode-extensions
ms-vscode-remote.remote-ssh
```

3. **Connect via SSH:**
- Open Command Palette (Ctrl+Shift+P)
- "Remote-SSH: Connect to Host"
- Enter: `ssh root@<ngrok_host> -p <ngrok_port>`
- Password: `password123`

### Pros/Cons:

**Method 1 (Direct Upload):**
✅ Simple and fast
✅ Full GPU acceleration
❌ Manual file sync

**Method 2 (GitHub Sync - Current):**
✅ Version controlled
✅ Automated sync
✅ Already working perfectly
❌ Requires internet for sync

**Method 3 (SSH Remote):**
✅ Native VS Code experience
✅ Direct file editing
❌ Complex setup
❌ Ngrok dependency
❌ Connection can be unstable

## Recommendation

**Stick with Method 2 (GitHub Sync)** - it's working perfectly for your project:

1. ✅ You get full VS Code editing experience locally
2. ✅ Version control with Git
3. ✅ Colab Pro GPU acceleration
4. ✅ Automatic sync via `git pull`
5. ✅ Reliable and fast

Your current setup is actually the optimal approach for ML projects!

## Quick Commands for Your Current Workflow:

```bash
# In VS Code terminal:
git add .
git commit -m "Update training configuration"
git push

# In Colab:
!git pull origin main
exec(open('colab_training.py').read())
```

This gives you the best of both worlds: professional development environment + powerful cloud GPU training.