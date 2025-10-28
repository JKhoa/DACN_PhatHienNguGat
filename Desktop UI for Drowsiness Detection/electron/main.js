const { app, BrowserWindow, ipcMain, session } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true,
      webSecurity: false,  // Disable web security to allow localhost requests
      allowRunningInsecureContent: true  // Allow HTTP requests
    },
    title: 'Hệ thống Phát hiện Ngủ gật - Classroom Monitoring'
  });

  // Load the React app
  const indexPath = path.join(__dirname, '..', 'dist', 'index.html');
  
  console.log('Loading React app from:', indexPath);
  mainWindow.loadFile(indexPath);

  // Open DevTools for debugging
  mainWindow.webContents.openDevTools();

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Start Python backend
function startPythonBackend() {
  const pythonPath = path.join(__dirname, '..', 'python-backend');
  const serverPath = path.join(pythonPath, 'server.py');
  const reqsPath = path.join(pythonPath, 'requirements.txt');
  
  console.log('Starting Python backend...');
  console.log('Python path:', pythonPath);
  console.log('Server path:', serverPath);
  console.log('Ensuring backend dependencies are installed...');

  // First, install backend requirements (no-op if already satisfied)
  const pipInstall = spawn('python', ['-m', 'pip', 'install', '-r', reqsPath], {
    cwd: pythonPath,
    stdio: ['pipe', 'pipe', 'pipe']
  });

  pipInstall.stdout.on('data', (data) => {
    console.log(`pip stdout: ${data}`);
  });

  pipInstall.stderr.on('data', (data) => {
    console.error(`pip stderr: ${data}`);
  });

  pipInstall.on('close', (code) => {
    console.log(`pip install exited with code ${code}`);
    // After install attempt, start backend server
    console.log('Starting Python server...');
    pythonProcess = spawn('python', [serverPath], {
      cwd: pythonPath,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
    });

    pythonProcess.on('error', (err) => {
      console.error('Failed to start Python process:', err);
    });

    // Wait a bit for Python to start, then test connection
    setTimeout(() => {
      console.log('Testing Python backend connection...');
      fetch('http://127.0.0.1:5000/api/cameras')
        .then(response => {
          console.log('Python backend is responding:', response.status);
        })
        .catch(error => {
          console.error('Python backend connection failed:', error);
        });
    }, 3000);
  });
}

// IPC handlers for camera control
ipcMain.handle('get-camera-list', async () => {
  try {
    const response = await fetch('http://127.0.0.1:5000/api/cameras');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching camera list:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('add-camera', async (event, cameraData) => {
  try {
    const response = await fetch('http://127.0.0.1:5000/api/camera/add', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(cameraData)
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error adding camera:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('start-camera', async (event, cameraId) => {
  try {
    const response = await fetch(`http://127.0.0.1:5000/api/camera/${cameraId}/start`, {
      method: 'POST'
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error starting camera:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('stop-camera', async (event, cameraId) => {
  try {
    const response = await fetch(`http://127.0.0.1:5000/api/camera/${cameraId}/stop`, {
      method: 'POST'
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error stopping camera:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('remove-camera', async (event, cameraId) => {
  try {
    const response = await fetch(`http://127.0.0.1:5000/api/camera/${cameraId}/remove`, {
      method: 'DELETE'
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error removing camera:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-system-stats', async () => {
  try {
    const response = await fetch('http://127.0.0.1:5000/api/system/stats');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching system stats:', error);
    return { success: false, error: error.message };
  }
});

app.whenReady().then(() => {
  // Disable web security for localhost requests
  app.commandLine.appendSwitch('--disable-web-security');
  app.commandLine.appendSwitch('--allow-running-insecure-content');
  app.commandLine.appendSwitch('--disable-features', 'VizDisplayCompositor');
  
  // Auto-allow media (camera/microphone) permission requests for getUserMedia
  try {
    const sess = session.defaultSession;
    if (sess && sess.setPermissionRequestHandler) {
      sess.setPermissionRequestHandler((webContents, permission, callback) => {
        if (permission === 'media' || permission === 'camera' || permission === 'microphone') {
          return callback(true);
        }
        return callback(true);
      });
    }
  } catch (e) {
    console.warn('Could not configure media permission handler:', e);
  }
  
  // Start Python backend first
  startPythonBackend();
  
  // Wait a bit for Python to start
  setTimeout(() => {
    createWindow();
  }, 2000);

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  // Kill Python process when app closes
  if (pythonProcess) {
    pythonProcess.kill();
  }
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  // Kill Python process before quitting
  if (pythonProcess) {
    pythonProcess.kill();
  }
});




