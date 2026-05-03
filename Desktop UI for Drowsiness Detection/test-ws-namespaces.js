/**
 * Test script to verify WebSocket namespace connections work correctly
 */
const { io } = require('socket.io-client');
const { spawn } = require('child_process');
const path = require('path');

const PYTHON_BASE = 'http://127.0.0.1:5000';
const DETECT_URL = `${PYTHON_BASE}/ws/detect`;
const CAMERA_URL = `${PYTHON_BASE}/ws/camera`;

let pythonProcess = null;
let detectSocket = null;
let cameraSocket = null;
let testsPassed = 0;
let testsFailed = 0;

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function log(msg) {
    console.log(`[TEST] ${msg}`);
}

function logSuccess(msg) {
    console.log(`✅ ${msg}`);
    testsPassed++;
}

function logError(msg) {
    console.error(`❌ ${msg}`);
    testsFailed++;
}

async function startPythonBackend() {
    log('Starting Python backend...');
    const pythonPath = path.join(__dirname, 'python-backend');
    const serverPath = path.join(pythonPath, 'server_with_tracking_backup.py');

    pythonProcess = spawn('python', [serverPath], {
        cwd: pythonPath,
        stdio: ['pipe', 'pipe', 'pipe']
    });

    pythonProcess.stdout.on('data', (data) => {
        const output = data.toString().trim();
        if (output.includes('Running on') || output.includes('uptime_s')) {
            log(`Python: ${output}`);
        }
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`[Python stderr] ${data}`);
    });

    pythonProcess.on('error', (err) => {
        logError(`Python process error: ${err.message}`);
    });

    // Wait for server to start
    await sleep(3000);
}

async function testHealthCheck() {
    log('Testing health check endpoint...');
    try {
        const response = await fetch(`${PYTHON_BASE}/api/health`, {
            signal: AbortSignal.timeout(5000)
        });
        if (response.ok) {
            const data = await response.json();
            logSuccess(`Health check OK: ${JSON.stringify(data)}`);
            return true;
        } else {
            logError(`Health check failed with status ${response.status}`);
            return false;
        }
    } catch (err) {
        logError(`Health check error: ${err.message}`);
        return false;
    }
}

async function testDetectNamespace() {
    log('Testing /ws/detect namespace connection...');

    return new Promise((resolve) => {
        try {
            detectSocket = io(DETECT_URL, {
                path: '/socket.io/',
                transports: ['polling'],  // Force polling only
                reconnection: false,
                reconnectionAttempts: 1,
            });

            const connectTimeout = setTimeout(() => {
                logError(`/ws/detect namespace connect timeout (10s)`);
                detectSocket?.disconnect();
                resolve(false);
            }, 10000);

            detectSocket.on('connect', () => {
                clearTimeout(connectTimeout);
                logSuccess(`/ws/detect namespace connected`);
                resolve(true);
            });

            detectSocket.on('connect_error', (err) => {
                clearTimeout(connectTimeout);
                logError(`/ws/detect connect_error: ${err.message}`);
                resolve(false);
            });

            detectSocket.on('error', (err) => {
                clearTimeout(connectTimeout);
                logError(`/ws/detect error: ${err}`);
                resolve(false);
            });
        } catch (err) {
            logError(`Exception creating detect socket: ${err.message}`);
            resolve(false);
        }
    });
}

async function testCameraNamespace() {
    log('Testing /ws/camera namespace connection...');

    return new Promise((resolve) => {
        try {
            cameraSocket = io(CAMERA_URL, {
                path: '/socket.io/',
                transports: ['polling'],  // Force polling only
                reconnection: false,
                reconnectionAttempts: 1,
            });

            const connectTimeout = setTimeout(() => {
                logError(`/ws/camera namespace connect timeout (10s)`);
                cameraSocket?.disconnect();
                resolve(false);
            }, 10000);

            cameraSocket.on('connect', () => {
                clearTimeout(connectTimeout);
                logSuccess(`/ws/camera namespace connected`);
                resolve(true);
            });

            cameraSocket.on('connect_error', (err) => {
                clearTimeout(connectTimeout);
                logError(`/ws/camera connect_error: ${err.message}`);
                resolve(false);
            });

            cameraSocket.on('error', (err) => {
                clearTimeout(connectTimeout);
                logError(`/ws/camera error: ${err}`);
                resolve(false);
            });
        } catch (err) {
            logError(`Exception creating camera socket: ${err.message}`);
            resolve(false);
        }
    });
}

async function cleanup() {
    log('Cleaning up...');

    if (detectSocket) {
        detectSocket.disconnect();
    }
    if (cameraSocket) {
        cameraSocket.disconnect();
    }
    if (pythonProcess) {
        pythonProcess.kill();
    }

    // Give processes time to shutdown
    await sleep(500);
}

async function runTests() {
    try {
        log('=== WebSocket Namespace Connection Test ===');

        await startPythonBackend();

        const healthOk = await testHealthCheck();
        if (!healthOk) {
            logError('Health check failed, cannot proceed with WebSocket tests');
            await cleanup();
            process.exit(1);
        }

        await sleep(1000);

        const detectOk = await testDetectNamespace();
        await sleep(500);

        const cameraOk = await testCameraNamespace();

        await cleanup();

        log('');
        log('=== Test Summary ===');
        log(`Passed: ${testsPassed}`);
        log(`Failed: ${testsFailed}`);

        if (testsFailed === 0) {
            log('✅ All tests passed!');
            process.exit(0);
        } else {
            log('❌ Some tests failed');
            process.exit(1);
        }
    } catch (err) {
        logError(`Test runner error: ${err.message}`);
        console.error(err);
        await cleanup();
        process.exit(1);
    }
}

runTests();
