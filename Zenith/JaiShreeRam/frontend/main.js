const { app, BrowserWindow, Menu, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const simpleGit = require('simple-git');

let mainWindow;
let fileWatchers = new Map();

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    icon: path.join(__dirname, 'assets/icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      sandbox: false,
      webSecurity: false
    },
    titleBarStyle: 'hidden',
    frame: false,
    backgroundColor: '#1e1e1e',
    show: false,
    trafficLightPosition: { x: 15, y: 12 }
  });

  mainWindow.loadFile('index.html');

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }

  createApplicationMenu();
}

function createApplicationMenu() {
  const isMac = process.platform === 'darwin';

  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'New File',
          accelerator: 'CmdOrCtrl+N',
          click: () => mainWindow.webContents.send('menu-action', 'new-file')
        },
        {
          label: 'New Folder',
          accelerator: 'CmdOrCtrl+Shift+N',
          click: () => mainWindow.webContents.send('menu-action', 'new-folder')
        },
        { type: 'separator' },
        {
          label: 'Open File',
          accelerator: 'CmdOrCtrl+O',
          click: openFile
        },
        {
          label: 'Open Folder',
          accelerator: 'CmdOrCtrl+Shift+O',
          click: openFolder
        },
        { type: 'separator' },
        {
          label: 'Save',
          accelerator: 'CmdOrCtrl+S',
          click: () => mainWindow.webContents.send('menu-action', 'save-file')
        },
        {
          label: 'Save As',
          accelerator: 'CmdOrCtrl+Shift+S',
          click: () => mainWindow.webContents.send('menu-action', 'save-file-as')
        },
        { type: 'separator' },
        {
          label: 'Run',
          accelerator: 'F5',
          click: () => mainWindow.webContents.send('menu-action', 'run-code')
        },
        { type: 'separator' },
        isMac ? { role: 'close' } : {
          label: 'Exit',
          accelerator: 'CmdOrCtrl+Q',
          click: () => app.quit()
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo', accelerator: 'CmdOrCtrl+Z' },
        { role: 'redo', accelerator: 'CmdOrCtrl+Y' },
        { type: 'separator' },
        { role: 'cut', accelerator: 'CmdOrCtrl+X' },
        { role: 'copy', accelerator: 'CmdOrCtrl+C' },
        { role: 'paste', accelerator: 'CmdOrCtrl+V' },
        { role: 'selectAll', accelerator: 'CmdOrCtrl+A' },
        { type: 'separator' },
        {
          label: 'Find',
          accelerator: 'CmdOrCtrl+F',
          click: () => mainWindow.webContents.send('menu-action', 'find')
        },
        {
          label: 'Replace',
          accelerator: 'CmdOrCtrl+H',
          click: () => mainWindow.webContents.send('menu-action', 'replace')
        },
        { type: 'separator' },
        {
          label: 'Toggle Comment',
          accelerator: 'CmdOrCtrl+/',
          click: () => mainWindow.webContents.send('menu-action', 'toggle-comment')
        },
        {
          label: 'Format Document',
          accelerator: 'Shift+Alt+F',
          click: () => mainWindow.webContents.send('menu-action', 'format-document')
        }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        {
          label: 'Toggle Sidebar',
          accelerator: 'CmdOrCtrl+B',
          click: () => mainWindow.webContents.send('window-action', 'toggle-sidebar')
        },
        {
          label: 'Toggle AI Panel',
          accelerator: 'CmdOrCtrl+Shift+I',
          click: () => mainWindow.webContents.send('window-action', 'toggle-ai-panel')
        },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn', accelerator: 'CmdOrCtrl+=' },
        { role: 'zoomOut', accelerator: 'CmdOrCtrl+-' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'AI',
      submenu: [
        {
          label: 'Generate Code',
          accelerator: 'CmdOrCtrl+Shift+G',
          click: () => mainWindow.webContents.send('ai-action', 'generate')
        },
        {
          label: 'Explain Code',
          accelerator: 'CmdOrCtrl+Shift+E',
          click: () => mainWindow.webContents.send('ai-action', 'explain')
        },
        {
          label: 'Debug Code',
          accelerator: 'CmdOrCtrl+Shift+D',
          click: () => mainWindow.webContents.send('ai-action', 'debug')
        },
        {
          label: 'Optimize Code',
          accelerator: 'CmdOrCtrl+Shift+O',
          click: () => mainWindow.webContents.send('ai-action', 'optimize')
        },
        { type: 'separator' },
        {
          label: 'Ask AI Assistant',
          accelerator: 'CmdOrCtrl+I',
          click: () => mainWindow.webContents.send('ai-action', 'chat')
        }
      ]
    },
    {
      label: 'Terminal',
      submenu: [
        {
          label: 'New Terminal',
          accelerator: 'Ctrl+`',
          click: () => mainWindow.webContents.send('terminal-action', 'new-terminal')
        },
        {
          label: 'Run Current File',
          accelerator: 'Ctrl+Shift+R',
          click: () => mainWindow.webContents.send('terminal-action', 'run-current')
        }
      ]
    },
    ...(isMac ? [{
      label: 'Window',
      role: 'window',
      submenu: [
        { role: 'minimize' },
        { role: 'zoom' },
        { type: 'separator' },
        { role: 'front' },
        { type: 'separator' },
        { role: 'window' }
      ]
    }] : []),
    {
      label: 'Help',
      role: 'help',
      submenu: [
        {
          label: 'Documentation',
          click: () => shell.openExternal('https://code.visualstudio.com/docs')
        },
        {
          label: 'Keyboard Shortcuts',
          accelerator: 'CmdOrCtrl+Shift+P',
          click: () => mainWindow.webContents.send('menu-action', 'shortcuts')
        },
        { type: 'separator' },
        {
          label: 'About',
          click: () => mainWindow.webContents.send('menu-action', 'about')
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

async function openFile() {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'All Files', extensions: ['*'] },
      { name: 'JavaScript', extensions: ['js', 'jsx', 'mjs'] },
      { name: 'TypeScript', extensions: ['ts', 'tsx'] },
      { name: 'HTML', extensions: ['html', 'htm'] },
      { name: 'CSS', extensions: ['css', 'scss', 'sass', 'less'] },
      { name: 'Python', extensions: ['py', 'pyw'] },
      { name: 'JSON', extensions: ['json'] },
      { name: 'Markdown', extensions: ['md', 'markdown'] }
    ]
  });

  if (!canceled && filePaths.length > 0) {
    try {
      const filePath = filePaths[0];
      const content = fs.readFileSync(filePath, 'utf-8');
      const fileName = path.basename(filePath);

      mainWindow.webContents.send('file-opened', {
        filePath,
        content,
        fileName,
        fileType: path.extname(filePath).substring(1)
      });
    } catch (error) {
      console.error('Error reading file:', error);
    }
  }
}


function buildFolderStructure(rootPath, files) {
  const structure = {
    name: path.basename(rootPath),
    path: rootPath,
    type: 'folder',
    open: true,
    children: []
  };

  const fileMap = new Map();

  files.forEach(file => {
    const parts = file.relativePath.split(path.sep);
    let currentLevel = structure.children;

    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i];
      let folder = currentLevel.find(item => item.name === part && item.type === 'folder');

      if (!folder) {
        folder = {
          name: part,
          path: path.join(rootPath, parts.slice(0, i + 1).join(path.sep)),
          type: 'folder',
          open: false,
          children: []
        };
        currentLevel.push(folder);
      }

      currentLevel = folder.children;
    }

    currentLevel.push({
      name: file.name,
      path: file.path,
      type: 'file',
      content: file.content,
      size: file.size,
      modified: file.modified,
      extension: file.extension
    });
  });

  structure.children.sort((a, b) => {
    if (a.type === 'folder' && b.type !== 'folder') return -1;
    if (a.type !== 'folder' && b.type === 'folder') return 1;
    return a.name.localeCompare(b.name);
  });

  return structure;
}


async function openFolder() {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory', 'multiSelections']
  });

  if (!canceled && filePaths.length > 0) {
    const folderPath = filePaths[0];

    try {
      const getAllFiles = (dir, fileList = []) => {
        const files = fs.readdirSync(dir);

        files.forEach(file => {
          const filePath = path.join(dir, file);
          const stat = fs.statSync(filePath);

          if (stat.isDirectory()) {
            fileList.push({
              name: file,
              path: filePath,
              type: 'folder',
              relativePath: path.relative(folderPath, filePath)
            });
            getAllFiles(filePath, fileList);
          } else {
            const ext = path.extname(file).toLowerCase();
            const allowedExtensions = ['.js', '.ts', '.jsx', '.tsx', '.html', '.htm',
              '.css', '.scss', '.sass', '.less', '.py', '.json',
              '.md', '.txt', '.xml', '.yaml', '.yml', '.java',
              '.c', '.cpp', '.h', '.hpp', '.cs', '.php', '.rb',
              '.go', '.rs', '.sql'];

            if (allowedExtensions.includes(ext) || file.startsWith('.')) {
              try {
                const content = fs.readFileSync(filePath, 'utf-8');
                fileList.push({
                  name: file,
                  path: filePath,
                  type: 'file',
                  extension: ext.substring(1),
                  relativePath: path.relative(folderPath, filePath),
                  content: content,
                  size: stat.size,
                  modified: stat.mtime
                });
              } catch (error) {
                console.error(`Error reading file ${filePath}:`, error);
              }
            }
          }
        });

        return fileList;
      };

      const files = getAllFiles(folderPath);

      mainWindow.webContents.send('folder-opened', {
        folderPath,
        files: files,
        folderStructure: buildFolderStructure(folderPath, files)
      });

      watchFolder(folderPath);

    } catch (error) {
      console.error('Error reading folder:', error);
      dialog.showErrorBox('Error', `Failed to open folder: ${error.message}`);
    }
  }
}

function watchFolder(folderPath) {
  if (fileWatchers.has(folderPath)) {
    fileWatchers.get(folderPath).close();
  }

  const watcher = fs.watch(folderPath, { recursive: true }, (eventType, filename) => {
    if (filename) {
      mainWindow.webContents.send('file-changed', {
        eventType,
        filename,
        folderPath
      });
    }
  });

  fileWatchers.set(folderPath, watcher);
}

ipcMain.handle('save-file', async (event, { filePath, content }) => {
  if (!filePath || filePath.startsWith('/new/') || filePath.startsWith('/default/')) {
    const { canceled, filePath: newFilePath } = await dialog.showSaveDialog(mainWindow, {
      defaultPath: 'untitled.js',
      filters: [
        { name: 'JavaScript', extensions: ['js'] },
        { name: 'TypeScript', extensions: ['ts'] },
        { name: 'HTML', extensions: ['html'] },
        { name: 'CSS', extensions: ['css'] },
        { name: 'Python', extensions: ['py'] },
        { name: 'JSON', extensions: ['json'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    });

    if (canceled) return;
    filePath = newFilePath;
  }

  try {
    fs.writeFileSync(filePath, content, 'utf-8');

    event.sender.send('file-saved', { filePath });

    return filePath;
  } catch (error) {
    console.error('Error saving file:', error);
    throw error;
  }
});

ipcMain.handle('create-file', async (event, { folderPath, fileName }) => {
  try {
    let targetPath = folderPath;

    if (!folderPath || folderPath === '/' || folderPath.startsWith('/new/') || folderPath.startsWith('/default/')) {
      const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory'],
        title: 'Select folder to create file in'
      });

      if (canceled) return;
      targetPath = filePaths[0];
    }

    const filePath = path.join(targetPath, fileName);
    fs.writeFileSync(filePath, '', 'utf-8');

    event.sender.send('file-created', {
      filePath,
      fileName,
      folderPath: targetPath
    });

    return filePath;
  } catch (error) {
    console.error('Error creating file:', error);
    throw error;
  }
});

ipcMain.handle('create-folder', async (event, { folderPath, folderName }) => {
  try {
    let targetPath = folderPath;

    if (!folderPath || folderPath === '/' || folderPath.startsWith('/new/') || folderPath.startsWith('/default/')) {
      const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory'],
        title: 'Select parent folder'
      });

      if (canceled) return;
      targetPath = filePaths[0];
    }

    const newFolderPath = path.join(targetPath, folderName);
    fs.mkdirSync(newFolderPath, { recursive: true });

    event.sender.send('folder-created', {
      folderPath: newFolderPath,
      folderName,
      parentPath: targetPath
    });

    return newFolderPath;
  } catch (error) {
    console.error('Error creating folder:', error);
    throw error;
  }
});

ipcMain.handle('rename-file', async (event, { oldPath, newName }) => {
  try {
    const dir = path.dirname(oldPath);
    const newPath = path.join(dir, newName);

    if (fs.existsSync(newPath)) {
      throw new Error('A file with that name already exists');
    }

    fs.renameSync(oldPath, newPath);

    event.sender.send('file-renamed', {
      oldPath,
      newPath,
      newName
    });

    return newPath;
  } catch (error) {
    console.error('Error renaming file:', error);
    throw error;
  }
});

ipcMain.handle('delete-file', async (event, { filePath }) => {
  try {
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      fs.rmdirSync(filePath, { recursive: true });
    } else {
      fs.unlinkSync(filePath);
    }

    event.sender.send('file-deleted', { filePath });

    return true;
  } catch (error) {
    console.error('Error deleting file:', error);
    throw error;
  }
});

ipcMain.handle('run-code', async (event, { filePath, language }) => {
  try {
    return {
      success: true,
      output: {
        stdout: 'Code executed successfully!\nThis is a demo output. In production, this would execute the actual code.',
        stderr: '',
        code: 0
      }
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('window-control', (event, action) => {
  switch (action) {
    case 'minimize':
      mainWindow.minimize();
      break;
    case 'maximize':
      if (mainWindow.isMaximized()) {
        mainWindow.unmaximize();
      } else {
        mainWindow.maximize();
      }
      break;
    case 'close':
      mainWindow.close();
      break;
  }
});

ipcMain.handle('get-directory-files', async (event, dirPath) => {
  try {
    if (!fs.existsSync(dirPath)) {
      return [];
    }

    const files = fs.readdirSync(dirPath, { withFileTypes: true });
    return files.map(dirent => ({
      name: dirent.name,
      path: path.join(dirPath, dirent.name),
      type: dirent.isDirectory() ? 'folder' : 'file',
      extension: dirent.isFile() ? path.extname(dirent.name).substring(1) : ''
    }));
  } catch (error) {
    console.error('Error reading directory:', error);
    return [];
  }
});

ipcMain.handle('agent-edit', async (event, data) => {
  try {
    const response = await fetch('http://127.0.0.1:5000/api/agent/edit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      const errorText = await response.text();
      return { success: false, error: `Backend error: ${response.status} - ${errorText}` };
    }

    return await response.json();
  } catch (error) {
    console.error('Error in agent-edit:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('read-file', async (event, filePath) => {
  try {
    return fs.readFileSync(filePath, 'utf-8');
  } catch (error) {
    console.error('Error reading file:', error);
    throw error;
  }
});

ipcMain.handle('git-action', async (event, { action, path: workspacePath, data }) => {
  if (!workspacePath) return { success: false, error: 'No workspace path' };

  const git = simpleGit(workspacePath);

  try {
    // Check if it's a git repo first
    const isRepo = await git.checkIsRepo();
    if (!isRepo && action !== 'init') {
      return { success: false, error: 'Not a git repository' };
    }

    switch (action) {
      case 'status': {
        const repoStatus = await git.status();
        return {
          success: true,
          branch: repoStatus.current,
          files: repoStatus.files.map(f => ({
            path: f.path,
            status: f.working_dir || f.index,
            isConflict: (f.working_dir === 'U' || f.index === 'U')
          })),
          isMerge: (await fs.promises.access(path.join(workspacePath, '.git', 'MERGE_HEAD')).then(() => true).catch(() => false))
        };
      }

      case 'log': {
        const log = await git.log({ n: 20 });
        return { success: true, all: log.all };
      }

      case 'branches': {
        const branches = await git.branch();
        return { success: true, all: branches.all, current: branches.current };
      }

      case 'create-branch': {
        if (data.stash) {
          await git.stash(['push', '-m', `Auto-stash before creating branch ${data.name}`]);
        }
        await git.checkoutLocalBranch(data.name);
        return { success: true, message: `Created and switched to branch ${data.name}` };
      }

      case 'checkout': {
        try {
          if (data.stash) {
            await git.stash(['push', '-m', `Auto-stash before checkout to ${data.name}`]);
          }
          await git.checkout(data.name);
          if (data.pop) {
            try { await git.stash(['pop']); } catch (e) { /* might conflict */ }
          }
          return { success: true, message: `Switched to branch ${data.name}` };
        } catch (e) {
          return { success: false, error: e.message };
        }
      }

      case 'stash': {
        await git.stash(['push', '-m', data.message || 'Manual stash']);
        return { success: true, message: 'Changes stashed' };
      }

      case 'stash-pop': {
        await git.stash(['pop']);
        return { success: true, message: 'Latest stash applied' };
      }

      case 'commit': {
        await git.add('.');
        const commitResult = await git.commit(data || 'Manual commit from Zenith');
        return { success: true, message: `Committed ${commitResult.commit}` };
      }

      case 'push':
      case 'pull':
      case 'sync': {
        const currentStatus = await git.status();
        let remoteUrl = '';
        try {
          remoteUrl = await git.remote(['get-url', 'origin']);
          remoteUrl = remoteUrl.trim();

          if (data && data.token && remoteUrl.includes('github.com')) {
            // Insert token into URL: https://TOKEN@github.com/user/repo.git
            if (remoteUrl.startsWith('https://')) {
              remoteUrl = remoteUrl.replace('https://', `https://${data.token}@`);
            } else if (remoteUrl.startsWith('git@github.com:')) {
              const match = remoteUrl.match(/git@github.com:(.*)\.git/);
              if (match) {
                remoteUrl = `https://${data.token}@github.com/${match[1]}.git`;
              }
            }
          }
        } catch (e) {
          return { success: false, error: 'No remote "origin" found. Please add a remote first.' };
        }

        if (action === 'pull' || action === 'sync') {
          await git.pull(remoteUrl, currentStatus.current);
        }
        if (action === 'push' || action === 'sync') {
          await git.push(remoteUrl, currentStatus.current);
        }
        return { success: true, message: `${action.toUpperCase()} successful` };
      }

      case 'init': {
        await git.init();
        return { success: true, message: 'Git repository initialized' };
      }

      default:
        return { success: false, error: `Unknown git action: ${action}` };
    }
  } catch (error) {
    console.error(`Git action ${action} error:`, error);
    return { success: false, error: error.message };
  }
});

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  fileWatchers.forEach(watcher => watcher.close());
  fileWatchers.clear();

  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('web-contents-created', (event, contents) => {
  contents.on('new-window', (event, navigationUrl) => {
    event.preventDefault();
    shell.openExternal(navigationUrl);
  });
});

ipcMain.handle('show-open-file-dialog', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'All Files', extensions: ['*'] },
      { name: 'JavaScript', extensions: ['js', 'jsx', 'mjs'] },
      { name: 'TypeScript', extensions: ['ts', 'tsx'] },
      { name: 'HTML', extensions: ['html', 'htm'] },
      { name: 'CSS', extensions: ['css', 'scss', 'sass', 'less'] },
      { name: 'Python', extensions: ['py', 'pyw'] },
      { name: 'JSON', extensions: ['json'] },
      { name: 'Markdown', extensions: ['md', 'markdown'] }
    ]
  });

  return { canceled, filePaths };
});

ipcMain.handle('show-open-folder-dialog', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory']
  });

  return { canceled, filePaths };
});

ipcMain.handle('get-file-stats', async (event, filePath) => {
  try {
    const stats = fs.statSync(filePath);
    return {
      size: stats.size,
      modified: stats.mtime,
      created: stats.birthtime,
      isDirectory: stats.isDirectory()
    };
  } catch (error) {
    console.error('Error getting file stats:', error);
    return null;
  }
});
ipcMain.on('file-changed-externally', (event, data) => {
  mainWindow.webContents.send('file-changed-externally', data);
});
ipcMain.handle('watch-file', async (event, filePath) => {
  try {
    const watcher = fs.watch(filePath, (eventType, filename) => {
      mainWindow.webContents.send('file-changed-externally', {
        filePath,
        eventType,
        filename
      });
    });

    return { success: true, watcherId: watcher.uid };
  } catch (error) {
    console.error('Error watching file:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('unwatch-file', async (event, filePath) => {
  return { success: true };
});