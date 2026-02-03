const { contextBridge, ipcRenderer } = require('electron');

const API_BASE_URL = 'http://127.0.0.1:5000/api';

contextBridge.exposeInMainWorld('electronAPI', {
    windowControl: (action) => ipcRenderer.invoke('window-control', action),
    invoke: (channel, data) => ipcRenderer.invoke(channel, data),
    
    saveFile: (data) => ipcRenderer.invoke('save-file', data),
    createFile: (data) => ipcRenderer.invoke('create-file', data),
    createFolder: (data) => ipcRenderer.invoke('create-folder', data),
    renameFile: (data) => ipcRenderer.invoke('rename-file', data),
    deleteFile: (data) => ipcRenderer.invoke('delete-file', data),
    getDirectoryFiles: (dirPath) => ipcRenderer.invoke('get-directory-files', dirPath),
    runCode: (data) => ipcRenderer.invoke('run-code', data),
    readFileContent: (filePath) => {
        return new Promise((resolve, reject) => {
            try {
                const fs = require('fs');
                const content = fs.readFileSync(filePath, 'utf-8');
                resolve(content);
            } catch (error) {
                reject(error);
            }
        });
    },

    resetRAGIndex: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/rag/reset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error resetting RAG index:', error);
            return {
                success: false,
                error: error.message
            };
        }
    },
    
    getFolderStructure: async (folderPath) => {
        return new Promise((resolve, reject) => {
            try {
                const fs = require('fs');
                const path = require('path');
                
                const readDirRecursive = (dir) => {
                    const items = [];
                    const files = fs.readdirSync(dir);
                    const sortedFiles = files.sort((a, b) => {
                        const statA = fs.statSync(path.join(dir, a));
                        const statB = fs.statSync(path.join(dir, b));
                        
                        if (statA.isDirectory() && !statB.isDirectory()) return -1;
                        if (!statA.isDirectory() && statB.isDirectory()) return 1;
                        return a.localeCompare(b);
                    });
                    
                    sortedFiles.forEach(file => {
                        const filePath = path.join(dir, file);
                        const stat = fs.statSync(filePath);
                        
                        if (stat.isDirectory()) {
                            items.push({
                                name: file,
                                path: filePath,
                                type: 'folder',
                                children: readDirRecursive(filePath),
                                open: false
                            });
                        } else {
                            const ext = path.extname(file).toLowerCase();
                            const textExtensions = ['.js', '.ts', '.jsx', '.tsx', '.html', '.htm', 
                                                  '.css', '.scss', '.sass', '.less', '.py', '.json', 
                                                  '.md', '.txt', '.xml', '.yaml', '.yml', '.java', 
                                                  '.c', '.cpp', '.h', '.hpp', '.cs', '.php', '.rb', 
                                                  '.go', '.rs', '.sql', '.vue', '.svelte'];
                            
                            if (textExtensions.includes(ext) || file.startsWith('.')) {
                                try {
                                    const content = fs.readFileSync(filePath, 'utf-8');
                                    items.push({
                                        name: file,
                                        path: filePath,
                                        type: 'file',
                                        content: content,
                                        size: stat.size,
                                        modified: stat.mtime,
                                        extension: ext.substring(1)
                                    });
                                } catch (error) {
                                    items.push({
                                        name: file,
                                        path: filePath,
                                        type: 'file',
                                        content: `// Binary file: ${file}\n// Size: ${stat.size} bytes`,
                                        size: stat.size,
                                        modified: stat.mtime,
                                        extension: ext.substring(1),
                                        isBinary: true
                                    });
                                }
                            }
                        }
                    });
                    
                    return items;
                };
                
                const structure = readDirRecursive(folderPath);
                resolve(structure);
            } catch (error) {
                reject(error);
            }
        });
    },
    
    showOpenFileDialog: async () => {
        return await ipcRenderer.invoke('show-open-file-dialog');
    },
    
    showOpenFolderDialog: async () => {
        return await ipcRenderer.invoke('show-open-folder-dialog');
    },
    
    getFileStats: async (filePath) => {
        return await ipcRenderer.invoke('get-file-stats', filePath);
    },
    
    watchFile: async (filePath) => {
        return await ipcRenderer.invoke('watch-file', filePath);
    },
    
    unwatchFile: async (filePath) => {
        return await ipcRenderer.invoke('unwatch-file', filePath);
    },
    
    watchFolder: async (folderPath) => {
        return new Promise((resolve, reject) => {
            try {
                const fs = require('fs');
                const path = require('path');
                const watcher = fs.watch(folderPath, { recursive: true }, (eventType, filename) => {
                    if (filename) {
                        const filePath = path.join(folderPath, filename);
                        ipcRenderer.send('file-changed-externally', {
                            eventType,
                            filename,
                            filePath
                        });
                    }
                });
                
                resolve({ success: true });
            } catch (error) {
                reject(error);
            }
        });
    },
    
    generateCode: async (prompt, context, language = 'python') => {
        try {
            const response = await fetch(`${API_BASE_URL}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    prompt, 
                    context: context || '',
                    language 
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error generating code:', error);
            return {
                success: false,
                error: error.message,
                code: `// Error: Could not connect to AI backend\n// ${error.message}\n\n// Prompt: ${prompt}\n// Try: python app.py in backend directory`
            };
        }
    },
    
    explainCode: async (code, language = 'auto') => {
        try {
            const response = await fetch(`${API_BASE_URL}/explain`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    code,
                    language 
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error explaining code:', error);
            return {
                success: false,
                error: error.message,
                explanation: `Error: ${error.message}\n\nMake sure the Flask backend is running on port 5000.`
            };
        }
    },
    
    debugCode: async (code, language = 'auto', errorMessage = '') => {
        try {
            const response = await fetch(`${API_BASE_URL}/debug`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    code,
                    language,
                    error_message: errorMessage 
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error debugging code:', error);
            return {
                success: false,
                error: error.message,
                debugged_code: code,
                explanation: `Debugging error: ${error.message}`
            };
        }
    },
    
    optimizeCode: async (code, language = 'auto', optimizationType = 'performance') => {
        try {
            const response = await fetch(`${API_BASE_URL}/optimize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    code,
                    language,
                    optimization_type: optimizationType 
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error optimizing code:', error);
            return {
                success: false,
                error: error.message,
                optimized_code: code,
                explanation: `Optimization error: ${error.message}`
            };
        }
    },
    
    chat: async (message, history = [], context = {}) => {
        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message,
                    history,
                    context 
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error chatting with AI:', error);
            return {
                success: false,
                error: error.message,
                response: `I'm having trouble connecting to the AI backend. Please make sure the Flask server is running.\n\nError: ${error.message}\n\nTo start the backend:\n1. Open terminal in backend directory\n2. Run: python app.py\n3. Make sure it's running on http://localhost:5000`,
                history: history
            };
        }
    },
    
    writeTests: async (code, language = 'auto', testFramework = '') => {
        try {
            const response = await fetch(`${API_BASE_URL}/test`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    code,
                    language,
                    test_framework: testFramework 
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error writing tests:', error);
            return {
                success: false,
                error: error.message,
                tests: `// Error writing tests: ${error.message}`,
                test_explanation: `Test generation failed. Please check backend connection.`
            };
        }
    },
    
    analyzeCode: async (code, language = 'auto', analysisType = 'comprehensive') => {
        try {
            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    code,
                    language,
                    analysis_type: analysisType 
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error analyzing code:', error);
            return {
                success: false,
                error: error.message,
                analysis: `Analysis error: ${error.message}`,
                quality_score: 0
            };
        }
    },
    
    convertCode: async (code, sourceLanguage = 'auto', targetLanguage = 'python') => {
        try {
            const response = await fetch(`${API_BASE_URL}/convert`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    code,
                    source_language: sourceLanguage,
                    target_language: targetLanguage 
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error converting code:', error);
            return {
                success: false,
                error: error.message,
                converted_code: `// Conversion error: ${error.message}`,
                explanation: `Could not convert code. Please check backend connection.`
            };
        }
    },
    
    documentCode: async (code, language = 'auto', documentationStyle = 'comprehensive') => {
        try {
            const response = await fetch(`${API_BASE_URL}/document`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    code,
                    language,
                    documentation_style: documentationStyle 
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error documenting code:', error);
            return {
                success: false,
                error: error.message,
                documented_code: code,
                documentation: `Documentation error: ${error.message}`
            };
        }
    },

    multiFileEdit: async (task, files) => {
        try {
            const response = await fetch(`${API_BASE_URL}/agent/edit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    task,
                    files
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error in multi-file edit:', error);
            return {
                success: false,
                error: error.message
            };
        }
    },
    
    checkBackendHealth: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return {
                success: true,
                status: data.status,
                service: data.service,
                version: data.version
            };
        } catch (error) {
            console.error('Backend health check failed:', error);
            return {
                success: false,
                error: error.message,
                status: 'unavailable',
                message: 'Flask backend is not running. Please start it with: python app.py in backend directory'
            };
        }
    },
    
    analyzeFiles: async (files, analysisType = 'structure') => {
        try {
            const formData = new FormData();
            files.forEach(file => {
                formData.append('files', file);
            });
            formData.append('analysis_type', analysisType);
            
            const response = await fetch(`${API_BASE_URL}/files/analyze`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error analyzing files:', error);
            return {
                success: false,
                error: error.message,
                analysis: `File analysis error: ${error.message}`
            };
        }
    },
    
    indexCodebase: async (path) => {
        try {
            const response = await fetch(`${API_BASE_URL}/rag/index`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error indexing codebase:', error);
            return {
                success: false,
                error: error.message
            };
        }
    },
    
    checkOllamaHealth: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/ollama/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error checking Ollama health:', error);
            return {
                success: false,
                ollama_running: false,
                error: error.message,
                message: 'Ollama is not running. Please start it with: ollama serve'
            };
        }
    },
    
    pullOllama: async (modelName = 'codellama:7b') => {
        try {
            const response = await fetch(`${API_BASE_URL}/ollama/pull`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: modelName })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error pulling Ollama model:', error);
            return {
                success: false,
                error: error.message,
                message: 'Failed to pull model'
            };
        }
    },
    
    onFileOpened: (callback) => {
        ipcRenderer.on('file-opened', (event, data) => callback(data));
    },
    
    onFolderOpened: (callback) => {
        ipcRenderer.on('folder-opened', (event, data) => callback(data));
    },
    
    onFileSaved: (callback) => {
        ipcRenderer.on('file-saved', (event, data) => callback(data));
    },
    
    onFileCreated: (callback) => {
        ipcRenderer.on('file-created', (event, data) => callback(data));
    },
    
    onFolderCreated: (callback) => {
        ipcRenderer.on('folder-created', (event, data) => callback(data));
    },
    
    onFileRenamed: (callback) => {
        ipcRenderer.on('file-renamed', (event, data) => callback(data));
    },
    
    onFileDeleted: (callback) => {
        ipcRenderer.on('file-deleted', (event, data) => callback(data));
    },
    
    onFileChanged: (callback) => {
        ipcRenderer.on('file-changed', (event, data) => callback(data));
    },
    
    onMenuAction: (callback) => {
        ipcRenderer.on('menu-action', (event, action) => callback(action));
    },
    
    onAIAction: (callback) => {
        ipcRenderer.on('ai-action', (event, action) => callback(action));
    },
    
    onWindowAction: (callback) => {
        ipcRenderer.on('window-action', (event, action) => callback(action));
    },
    
    onTerminalAction: (callback) => {
        ipcRenderer.on('terminal-action', (event, action) => callback(action));
    },
    
    onFileChangedExternally: (callback) => {
        ipcRenderer.on('file-changed-externally', (event, data) => callback(data));
    },
    
    sendEditorContent: (content) => {
        ipcRenderer.send('editor-content', content);
    }
});

console.log('Preload script loaded');