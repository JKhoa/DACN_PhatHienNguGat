/**
 * Preload script — IPC bridge between main process and renderer.
 * Exposes window.appApi via contextBridge so the renderer has NO
 * direct access to Node.js or Electron internals.
 */
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('appApi', {
    /**
     * Invoke an IPC handler in the main process and return its result.
     * Usage: window.appApi.invoke('api:request', { method, endpoint, data })
     */
    invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),

    /**
     * Listen for push events sent from main via mainWindow.webContents.send().
     * Returns an unsubscribe function.
     * Usage (future WS bridge):
     *   const unsub = window.appApi.on('ws:result', (data) => { ... });
     *   unsub(); // cleanup
     */
    on: (channel, callback) => {
        const listener = (_event, ...args) => callback(...args);
        ipcRenderer.on(channel, listener);
        return () => ipcRenderer.removeListener(channel, listener);
    },

    /**
     * Remove all listeners for a channel.
     */
    removeAllListeners: (channel) => {
        ipcRenderer.removeAllListeners(channel);
    },
});
