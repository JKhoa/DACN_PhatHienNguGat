const { io } = require('socket.io-client');

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function test() {
    console.log('Testing minimal namespace...');

    const socket = io('http://127.0.0.1:5002/test', {
        path: '/socket.io/',
        transports: ['polling'],  // Try polling only first
        reconnection: false,
    });

    return new Promise((resolve) => {
        const timeout = setTimeout(() => {
            console.log('❌ Timeout - namespace connection failed');
            socket.disconnect();
            resolve(false);
        }, 5000);

        socket.on('connect', () => {
            clearTimeout(timeout);
            console.log('✅ Connected to /test namespace');
            socket.emit('test', { data: 'hello' });
        });

        socket.on('test_response', (msg) => {
            clearTimeout(timeout);
            console.log('✅ Received test_response:', msg);
            socket.disconnect();
            resolve(true);
        });

        socket.on('connect_error', (err) => {
            clearTimeout(timeout);
            console.log('❌ connect_error:', err.message);
            resolve(false);
        });

        socket.on('error', (err) => {
            clearTimeout(timeout);
            console.log('❌ error:', err);
            resolve(false);
        });
    });
}

(async () => {
    const success = await test();
    process.exit(success ? 0 : 1);
})();
