document.addEventListener('DOMContentLoaded', () => {
    window.codeEditor = new CodeEditor();
});

class CodeEditor {
    constructor() {
        this.editor = null;
        this.currentFile = null;
        this.workspacePath = null;
        this.fileTree = {};
        this.tabs = new Map();
        this.activeTab = null;
        this.aiPanelVisible = true;
        this.terminalVisible = false;
        this.aiChatHistory = [];
        this.renameContext = null;
        this.fileWatchers = new Map();
        this.fileSystemWatchers = new Map();
        this.autoSaveEnabled = true;
        this.isWindows = navigator.platform.indexOf('Win') > -1;

        this.isWindows = navigator.platform.indexOf('Win') > -1;
        this.pendingEdits = []; // Store edits for review
        this.diffEditor = null; // Monaco Diff Editor instance

        this.init();
    }

    async init() {
        await this.loadMonacoEditor();
        this.setupEventListeners();
        this.setupElectronHandlers();
        this.setupDefaultWorkspace();
        this.checkBackendHealth();
        this.setupAutoSave();
        this.setupFileSystemWatchers();
    }

    async loadMonacoEditor() {
        return new Promise((resolve) => {
            if (typeof monaco === 'undefined') {
                console.error('Monaco Editor not loaded');
                setTimeout(() => this.loadMonacoEditor().then(resolve), 100);
                return;
            }

            try {
                this.editor = monaco.editor.create(document.getElementById('editor'), {
                    value: '',
                    language: 'javascript',
                    theme: 'vs-dark',
                    automaticLayout: true,
                    minimap: {
                        enabled: true,
                        size: 'proportional'
                    },
                    fontSize: 14,
                    lineNumbers: 'on',
                    roundedSelection: false,
                    scrollBeyondLastLine: false,
                    readOnly: false,
                    wordWrap: 'on',
                    folding: true,
                    lineDecorationsWidth: 10,
                    lineNumbersMinChars: 3,
                    glyphMargin: true,
                    scrollbar: {
                        vertical: 'visible',
                        horizontal: 'visible',
                        useShadows: false,
                        verticalScrollbarSize: 10,
                        horizontalScrollbarSize: 10
                    }
                });

                // Add editor event listeners
                this.editor.onDidChangeModelContent(() => {
                    this.updateOutline();
                    this.updateTabStatus();
                });

                this.editor.onDidChangeCursorPosition((e) => {
                    const position = e.position;
                    document.getElementById('cursor-position').textContent =
                        `Ln ${position.lineNumber}, Col ${position.column}`;
                });

                resolve();
            } catch (error) {
                console.error('Failed to initialize Monaco Editor:', error);
                setTimeout(() => this.loadMonacoEditor().then(resolve), 100);
            }
        });
    }

    setupAutoSave() {
        if (this.editor) {
            this.editor.onDidChangeModelContent(() => {
                if (this.autoSaveEnabled && this.currentFile && !this.currentFile.startsWith('/')) {
                    this.debouncedSave();
                }
            });
        }

        this.debouncedSave = this.debounce(() => {
            if (this.currentFile && !this.currentFile.startsWith('/')) {
                this.saveCurrentFile();
            }
        }, 2000);
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    setupFileSystemWatchers() {
        window.electronAPI.onFileChanged((data) => {
            if (data.eventType === 'change' && this.currentFile && this.currentFile.includes(data.filename)) {
                this.handleExternalFileChange(this.currentFile);
            }
        });

        window.electronAPI.onFileChangedExternally((data) => {
            this.handleExternalFileChange(data.filePath);
        });
    }

    setupEventListeners() {
        console.log('Setting up event listeners...');

        document.getElementById('window-close').addEventListener('click', () => {
            window.electronAPI.windowControl('close');
        });

        document.getElementById('window-minimize').addEventListener('click', () => {
            window.electronAPI.windowControl('minimize');
        });

        document.getElementById('window-maximize').addEventListener('click', () => {
            window.electronAPI.windowControl('maximize');
        });

        document.querySelectorAll('.menu-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const menu = e.target.dataset.menu;
                this.showMenu(menu, e.target);
            });
        });

        document.getElementById('run-btn').addEventListener('click', () => {
            this.runCurrentFile();
        });

        // AI Toggle Button
        document.getElementById('ai-toggle-btn').addEventListener('click', () => {
            this.toggleAIPanel();
        });

        // Sidebar Actions
        document.getElementById('new-file-btn').addEventListener('click', () => {
            this.createNewFile();
        });

        document.getElementById('new-folder-btn').addEventListener('click', () => {
            this.createNewFolder();
        });

        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.refreshFileTree();
        });

        // Tab Actions
        document.getElementById('split-editor-btn').addEventListener('click', () => {
            this.splitEditor();
        });

        document.getElementById('close-all-tabs-btn').addEventListener('click', () => {
            this.closeAllTabs();
        });

        // Terminal
        document.getElementById('close-terminal-btn').addEventListener('click', () => {
            this.toggleTerminal();
        });

        document.getElementById('clear-terminal-btn').addEventListener('click', () => {
            this.clearTerminal();
        });

        document.getElementById('new-terminal-btn').addEventListener('click', () => {
            this.newTerminal();
        });

        const terminalInput = document.getElementById('terminal-input');
        if (terminalInput) {
            terminalInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.executeTerminalCommand(e.target.value);
                    e.target.value = '';
                }
            });
        }

        // AI Panel
        document.getElementById('close-ai-panel').addEventListener('click', () => {
            this.toggleAIPanel();
        });

        document.getElementById('ai-settings-btn').addEventListener('click', () => {
            this.showAISettings();
        });

        // RAG Toggle Logic
        const ragToggle = document.getElementById('rag-toggle');
        if (ragToggle) {
            ragToggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.indexCodebase();
                }
            });
        }

        // AI Quick Actions
        document.querySelectorAll('.ai-action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.handleAIAction(action);
            });
        });

        // AI Chat
        document.getElementById('send-ai-message').addEventListener('click', () => {
            this.sendAIMessage();
        });

        document.getElementById('clear-chat-btn').addEventListener('click', () => {
            this.clearAIChat();
        });

        document.getElementById('attach-file-btn').addEventListener('click', () => {
            this.attachFileToChat();
        });

        const aiInput = document.getElementById('ai-input');
        if (aiInput) {
            aiInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendAIMessage();
                }
            });
        }

        // Context Menu
        document.addEventListener('contextmenu', (e) => {
            if (e.target.closest('.file-item, .folder-item')) {
                e.preventDefault();
                this.showFileContextMenu(e);
            } else if (e.target.closest('#editor')) {
                e.preventDefault();
                this.showEditorContextMenu(e);
            }
        });

        // Close context menus on click
        document.addEventListener('click', () => {
            this.hideContextMenus();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });

        // Initialize tabs container
        this.setupTabListeners();

        // Rename dialog
        document.getElementById('rename-cancel').addEventListener('click', () => {
            this.hideRenameDialog();
        });

        document.getElementById('rename-confirm').addEventListener('click', () => {
            this.confirmRename();
        });

        // New file dialog
        document.getElementById('new-file-cancel').addEventListener('click', () => {
            this.hideNewFileDialog();
        });

        document.getElementById('new-file-confirm').addEventListener('click', () => {
            this.confirmNewFile();
        });

        // Handle Enter key in new file input
        document.getElementById('new-file-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.confirmNewFile();
            }
        });

        // New folder dialog
        document.getElementById('new-folder-cancel').addEventListener('click', () => {
            this.hideNewFolderDialog();
        });

        document.getElementById('new-folder-confirm').addEventListener('click', () => {
            this.confirmNewFolder();
        });

        // Handle Enter key in new folder input
        document.getElementById('new-folder-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.confirmNewFolder();
            }
        });

        // Open File/Folder buttons
        const openFileBtn = document.createElement('button');
        openFileBtn.className = 'action-btn btn-modern';
        openFileBtn.innerHTML = '<i class="fas fa-file"></i><span>Open File</span>';
        openFileBtn.addEventListener('click', () => this.openFileDialog());

        const openFolderBtn = document.createElement('button');
        openFolderBtn.className = 'action-btn btn-modern';
        openFolderBtn.innerHTML = '<i class="fas fa-folder-open"></i><span>Open Folder</span>';
        openFolderBtn.addEventListener('click', () => this.openFolderDialog());

        const titleBarActions = document.querySelector('.title-bar-actions');
        if (titleBarActions) {
            titleBarActions.insertBefore(openFileBtn, titleBarActions.firstChild);
            titleBarActions.insertBefore(openFolderBtn, titleBarActions.firstChild);
        }

        // Activity Bar
        document.querySelectorAll('.activity-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const panel = e.currentTarget.dataset.panel;
                if (panel) {
                    this.showPanel(panel);
                }
            });
        });

        // Git Actions
        const gitCommitBtn = document.getElementById('git-commit-btn');
        if (gitCommitBtn) {
            gitCommitBtn.addEventListener('click', () => {
                const message = document.getElementById('git-commit-message').value;
                this.handleGitAction('commit', message);
            });
        }

        const gitPushBtn = document.getElementById('git-push-btn');
        if (gitPushBtn) {
            gitPushBtn.addEventListener('click', () => this.handleGitAction('push'));
        }

        const gitPullBtn = document.getElementById('git-pull-btn');
        if (gitPullBtn) {
            gitPullBtn.addEventListener('click', () => this.handleGitAction('pull'));
        }

        const gitSyncBtn = document.getElementById('git-sync-btn');
        if (gitSyncBtn) {
            gitSyncBtn.addEventListener('click', () => this.handleGitAction('sync'));
        }

        const gitInitBtn = document.getElementById('git-init-btn');
        if (gitInitBtn) {
            gitInitBtn.addEventListener('click', () => this.handleGitAction('init'));
        }

        const gitNewBranchBtn = document.getElementById('git-new-branch-btn');
        if (gitNewBranchBtn) {
            gitNewBranchBtn.addEventListener('click', () => {
                document.getElementById('new-branch-dialog').style.display = 'flex';
                document.getElementById('new-branch-input').focus();
            });
        }

        const newBranchCancel = document.getElementById('new-branch-cancel');
        if (newBranchCancel) {
            newBranchCancel.addEventListener('click', () => {
                document.getElementById('new-branch-dialog').style.display = 'none';
                document.getElementById('new-branch-input').value = '';
            });
        }

        const newBranchConfirm = document.getElementById('new-branch-confirm');
        if (newBranchConfirm) {
            newBranchConfirm.addEventListener('click', () => {
                const name = document.getElementById('new-branch-input').value.trim();
                if (name) {
                    this.checkChangesAndPerform('create-branch', name);
                    document.getElementById('new-branch-dialog').style.display = 'none';
                    document.getElementById('new-branch-input').value = '';
                }
            });
        }

        const gitRefreshHistory = document.getElementById('git-refresh-history');
        if (gitRefreshHistory) {
            gitRefreshHistory.addEventListener('click', () => this.updateGitHistory());
        }

        const gitAIResolveBtn = document.getElementById('git-ai-resolve-btn');
        if (gitAIResolveBtn) {
            gitAIResolveBtn.addEventListener('click', () => this.resolveGitConflictsWithAI());
        }

        // Checkout Dialog Listeners
        const checkoutBring = document.getElementById('checkout-bring');
        if (checkoutBring) {
            checkoutBring.addEventListener('click', () => {
                const { action, branch } = checkoutBring.dataset;
                this.performBranchAction(action, branch, { stash: false });
                document.getElementById('git-checkout-dialog').style.display = 'none';
            });
        }

        const checkoutStash = document.getElementById('checkout-stash');
        if (checkoutStash) {
            checkoutStash.addEventListener('click', () => {
                const { action, branch } = checkoutStash.dataset;
                this.performBranchAction(action, branch, { stash: true });
                document.getElementById('git-checkout-dialog').style.display = 'none';
            });
        }

        const checkoutCancel = document.getElementById('checkout-cancel');
        if (checkoutCancel) {
            checkoutCancel.addEventListener('click', () => {
                document.getElementById('git-checkout-dialog').style.display = 'none';
            });
        }

        // Git Configuration
        const gitConfigBtn = document.getElementById('git-config-btn');
        if (gitConfigBtn) {
            gitConfigBtn.addEventListener('click', () => {
                document.getElementById('git-main-view').style.display = 'none';
                document.getElementById('git-no-repo-view').style.display = 'none';
                document.getElementById('git-config-view').style.display = 'block';

                // Load existing token
                const token = localStorage.getItem('github_token');
                if (token) {
                    document.getElementById('git-token-input').value = token;
                }
            });
        }

        const saveGitConfigBtn = document.getElementById('save-git-config');
        if (saveGitConfigBtn) {
            saveGitConfigBtn.addEventListener('click', () => {
                const token = document.getElementById('git-token-input').value;
                localStorage.setItem('github_token', token);
                this.showNotification('Git credentials saved', 'success');
                this.updateGitStatus(); // Redraw view
            });
        }

        const closeGitConfigBtn = document.getElementById('close-git-config');
        if (closeGitConfigBtn) {
            closeGitConfigBtn.addEventListener('click', () => {
                this.updateGitStatus(); // This will auto-switch to correct view
            });
        }

        // Documentation Actions
        const genProjectDocs = document.getElementById('gen-project-docs');
        if (genProjectDocs) {
            genProjectDocs.addEventListener('click', () => this.generateDocumentation('project'));
        }

        const genFileDocs = document.getElementById('gen-file-docs');
        if (genFileDocs) {
            genFileDocs.addEventListener('click', () => this.generateDocumentation('file'));
        }

        // Marketplace Search
        const marketplaceInput = document.querySelector('#marketplace-panel input');
        if (marketplaceInput) {
            // Debounce search
            let timeout;
            marketplaceInput.addEventListener('input', (e) => {
                clearTimeout(timeout);
                timeout = setTimeout(() => this.searchExtensions(e.target.value), 500);
            });
        }

        // Live Preview Actions
        const livePreviewBtn = document.getElementById('live-preview-btn');
        if (livePreviewBtn) {
            livePreviewBtn.addEventListener('click', () => this.toggleLivePreview());
        }

        const refreshPreview = document.getElementById('refresh-preview');
        if (refreshPreview) {
            refreshPreview.addEventListener('click', () => this.updateLivePreview());
        }

        const openExternalPreview = document.getElementById('open-external-preview');
        if (openExternalPreview) {
            openExternalPreview.addEventListener('click', () => {
                // Open the preview in a real browser
                const content = this.editor.getValue();
                const blob = new Blob([content], { type: 'text/html' });
                const url = URL.createObjectURL(blob);
                window.open(url, '_blank');
            });
        }

        console.log('Event listeners setup complete');
    }

    showPanel(panelId) {
        console.log(`Switching to panel: ${panelId}`);
        // Update activity bar
        document.querySelectorAll('.activity-item').forEach(item => {
            if (item.dataset.panel === panelId) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        // Update sidebar panels
        document.querySelectorAll('.sidebar-panel').forEach(panel => {
            if (panel.id === `${panelId}-panel`) {
                panel.style.display = 'flex';
                panel.classList.add('active');

                // Special handling for panels if needed
                if (panelId === 'git') {
                    this.updateGitStatus();
                } else if (panelId === 'marketplace') {
                    // Load popular extensions if list is empty or dummy
                    const list = document.getElementById('marketplace-list');
                    if (list && list.children.length > 0 && list.children[0].textContent.includes('Turbo Context')) {
                        this.searchExtensions(''); // Load default/popular
                    }
                }
            } else {
                panel.style.display = 'none';
                panel.classList.remove('active');
            }
        });
    }

    async handleGitAction(action, data = null) {
        console.log(`Git action: ${action}`, data);

        // Check for token on push/pull/sync
        if (['push', 'pull', 'sync'].includes(action)) {
            const token = localStorage.getItem('github_token');
            if (!token) {
                this.addAIMessage('ai', `⚠️ GitHub Token required for ${action}. Please click the 🔑 icon in Source Control to set it.`);
                this.showNotification(`GitHub Token required for ${action}`, 'warning');
                return;
            }
            data = { token }; // Pass token in data
        }

        this.addAIMessage('ai', `🌐 Performing Git ${action}...`);

        try {
            // Check if directory is a git repo
            const status = await window.electronAPI.invoke('git-action', { action: 'status', path: this.workspacePath });

            if (!status.success && action !== 'init') {
                this.addAIMessage('ai', `❌ Not a git repository. Would you like to initialize one?`);
                return;
            }

            const result = await window.electronAPI.invoke('git-action', {
                action,
                data,
                path: this.workspacePath
            });

            if (result.success) {
                this.addAIMessage('ai', `✅ Git ${action} successful! ${result.message || ''}`);
                this.updateGitStatus();
                this.showNotification(`Git ${action} successful`, 'success');
            } else {
                this.addAIMessage('ai', `❌ Git ${action} failed: ${result.error}`);
                this.showNotification(`Git ${action} failed`, 'error');
            }
        } catch (error) {
            console.error(`Git ${action} error:`, error);
            this.addAIMessage('ai', `❌ Error performing Git action: ${error.message}`);
        }
    }

    async updateGitStatus() {
        if (!this.workspacePath) return;

        // Reset views
        document.getElementById('git-config-view').style.display = 'none';

        try {
            const result = await window.electronAPI.invoke('git-action', {
                action: 'status',
                path: this.workspacePath
            });

            if (result.success) {
                document.getElementById('git-main-view').style.display = 'block';
                document.getElementById('git-no-repo-view').style.display = 'none';

                const branchStatus = document.getElementById('branch-status');
                if (branchStatus) {
                    branchStatus.querySelector('span').textContent = result.branch || 'main';
                }

                const changeCount = document.getElementById('git-change-count');
                if (changeCount) {
                    changeCount.textContent = result.files ? result.files.length : 0;
                }

                // Update Changes List
                const changesList = document.getElementById('git-changes-list');
                if (changesList) {
                    if (result.files && result.files.length > 0) {
                        changesList.innerHTML = result.files.map(file => `
                            <div class="git-list-item">
                                <i class="fas ${file.isConflict ? 'fa-exclamation-triangle' : 'fa-file'}"></i>
                                <span>${file.path}</span>
                                <span class="git-badge ${file.isConflict ? 'conflict' : ''}">${file.status}</span>
                            </div>
                        `).join('');
                    } else {
                        changesList.innerHTML = '<div class="empty-state">No changes detected</div>';
                    }
                }

                // Handle Conflicts
                const conflictSection = document.getElementById('git-conflict-section');
                const conflictList = document.getElementById('git-conflict-list');
                const conflicts = result.files ? result.files.filter(f => f.isConflict) : [];

                if (conflicts.length > 0) {
                    conflictSection.style.display = 'block';
                    conflictList.innerHTML = conflicts.map(f => `
                        <div class="git-list-item conflict">
                            <i class="fas fa-exclamation-circle" style="color: #f14c4c;"></i>
                            <span>${f.path}</span>
                        </div>
                    `).join('');
                } else {
                    conflictSection.style.display = 'none';
                }

                // Update branches and history
                this.updateGitBranches();
                this.updateGitHistory();
            } else {
                // Not a repo
                document.getElementById('git-main-view').style.display = 'none';
                document.getElementById('git-no-repo-view').style.display = 'block';
            }
        } catch (error) {
            console.error('Error updating git status:', error);
        }
    }

    async updateGitBranches() {
        if (!this.workspacePath) return;
        try {
            const result = await window.electronAPI.invoke('git-action', { action: 'branches', path: this.workspacePath });
            if (result.success) {
                const branchList = document.getElementById('git-branch-list');
                if (branchList) {
                    branchList.innerHTML = result.all.map(b => `
                        <div class="git-list-item ${b === result.current ? 'active' : ''}" data-branch="${b}" style="cursor: pointer;">
                            <i class="fas fa-code-branch"></i>
                            <span>${b}</span>
                            ${b === result.current ? '<span class="git-badge">current</span>' : ''}
                        </div>
                    `).join('');

                    // Add click listeners
                    branchList.querySelectorAll('.git-list-item').forEach(item => {
                        item.onclick = () => this.checkoutBranch(item.dataset.branch);
                    });
                }
            }
        } catch (error) {
            console.error('Error fetching branches:', error);
        }
    }

    async checkoutBranch(name) {
        this.checkChangesAndPerform('checkout', name);
    }

    async checkChangesAndPerform(action, branchName) {
        // Check for uncommitted changes first
        const statusResult = await window.electronAPI.invoke('git-action', { action: 'status', path: this.workspacePath });
        if (statusResult.success && statusResult.files && statusResult.files.length > 0) {
            const dialog = document.getElementById('git-checkout-dialog');
            const bringBtn = document.getElementById('checkout-bring');
            const stashBtn = document.getElementById('checkout-stash');

            bringBtn.dataset.action = action;
            bringBtn.dataset.branch = branchName;
            stashBtn.dataset.action = action;
            stashBtn.dataset.branch = branchName;

            dialog.style.display = 'flex';
            return;
        }

        // No changes, just perform action
        this.performBranchAction(action, branchName);
    }

    async performBranchAction(action, name, options = {}) {
        const displayAction = action === 'create-branch' ? 'Creating' : 'Switching to';
        this.addAIMessage('ai', `🌐 ${displayAction} branch ${name}...`);

        const result = await window.electronAPI.invoke('git-action', {
            action,
            path: this.workspacePath,
            data: { name, ...options }
        });

        if (result.success) {
            this.showNotification(result.message, 'success');
            this.updateGitStatus();
        } else {
            this.showNotification(`${action} failed: ${result.error}`, 'error');
            this.addAIMessage('ai', `❌ ${action} failed: ${result.error}`);
        }
    }

    async updateGitHistory() {
        if (!this.workspacePath) return;
        try {
            const result = await window.electronAPI.invoke('git-action', { action: 'log', path: this.workspacePath });
            if (result.success) {
                const historyList = document.getElementById('git-history-list');
                if (historyList) {
                    historyList.innerHTML = (result.all || []).map((commit, index) => `
                        <div class="git-history-item" title="${commit.message}">
                            <div class="git-graph-dot">
                                <div class="dot"></div>
                                ${index < (result.all || []).length - 1 ? '<div class="line"></div>' : ''}
                            </div>
                            <div class="git-history-details">
                                <div class="git-commit-msg">${commit.message}</div>
                                <div class="git-commit-meta">
                                    <span>${commit.author_name}</span>
                                    <span class="git-hash">${commit.hash.substring(0, 7)}</span>
                                </div>
                            </div>
                        </div>
                    `).join('');
                }
            }
        } catch (error) {
            console.error('Error fetching history:', error);
        }
    }

    async resolveGitConflictsWithAI() {
        this.addAIMessage('ai', '🤖 Analyzing merge conflicts for resolution...');
        const result = await window.electronAPI.invoke('git-action', { action: 'status', path: this.workspacePath });
        const conflicts = result.files.filter(f => f.isConflict);

        if (conflicts.length === 0) {
            this.addAIMessage('ai', 'No conflicts found.');
            return;
        }

        for (const file of conflicts) {
            try {
                // Read the file with conflict markers
                const content = await window.electronAPI.invoke('read-file', `${this.workspacePath}/${file.path}`);

                this.addAIMessage('ai', `Resolving ${file.path}...`);

                const response = await fetch('http://127.0.0.1:5000/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: `Merge conflict in file: ${file.path}.\nHere is the content with markers:\n\n${content}\n\nPlease resolve the conflict and provide ONLY the clean resolved code.`,
                        context: { task: 'resolve_conflict', file: file.path }
                    })
                });

                const data = await response.json();
                if (data.success) {
                    const resolvedCode = this.extractCodeBlock(data.response);
                    // Save the resolved code
                    await window.electronAPI.invoke('save-file', { filePath: `${this.workspacePath}/${file.path}`, content: resolvedCode });
                    this.addAIMessage('ai', `✅ Resolved conflict in ${file.path}`);
                }
            } catch (error) {
                this.addAIMessage('ai', `❌ Error resolving ${file.path}: ${error.message}`);
                console.error(`Error resolving conflict in ${file.path}:`, error);
            }
        }

        this.updateGitStatus();
    }

    extractCodeBlock(text) {
        if (!text) return "";
        const match = text.match(/```(?:\w+)?\n([\s\S]*?)\n```/);
        return match ? match[1] : text.trim();
    }

    setupTabListeners() {
        const tabsContainer = document.getElementById('tabs-container');
        if (tabsContainer) {
            tabsContainer.addEventListener('click', (e) => {
                const tab = e.target.closest('.tab');
                if (tab) {
                    if (e.target.classList.contains('tab-close')) {
                        e.stopPropagation();
                        const filePath = tab.dataset.file;
                        this.closeTab(filePath);
                    } else {
                        const filePath = tab.dataset.file;
                        this.setActiveTab(filePath);
                    }
                }
            });
        }
    }

    setupElectronHandlers() {
        console.log('Setting up Electron handlers...');

        if (!window.electronAPI) {
            console.error('electronAPI is not available');
            this.showNotification('Electron API not available', 'error');
            return;
        }

        // File operations
        window.electronAPI.onFileOpened((data) => {
            console.log('File opened:', data);
            this.openFile(data.filePath, data.content, data.fileName);
        });

        window.electronAPI.onFolderOpened((data) => {
            console.log('Folder opened:', data);
            this.openFolder(data.folderPath, data.files, data.folderStructure);

            // Auto-Index Codebase for RAG
            this.addAIMessage('ai', `📂 New folder detected. Indexing codebase for RAG context...`);
            this.indexCodebase(); // Uses current workspacePath set by openFolder
        });

        window.electronAPI.onFileSaved((data) => {
            console.log('File saved:', data);
            this.showNotification(`File saved: ${data.filePath}`, 'success');
            this.refreshFileTree();
        });

        window.electronAPI.onFileCreated((data) => {
            console.log('File created:', data);
            this.showNotification(`File created: ${data.fileName}`, 'success');
            this.refreshFileTree();
        });

        window.electronAPI.onFolderCreated((data) => {
            console.log('Folder created:', data);
            this.showNotification(`Folder created: ${data.folderName}`, 'success');
            this.refreshFileTree();
        });

        window.electronAPI.onFileRenamed((data) => {
            console.log('File renamed:', data);
            this.showNotification(`Renamed to: ${data.newName}`, 'success');
            this.refreshFileTree();

            // Update tab if it exists
            if (this.tabs.has(data.oldPath)) {
                const tab = this.tabs.get(data.oldPath);
                this.tabs.delete(data.oldPath);
                this.tabs.set(data.newPath, tab);

                // Update tab element
                const tabElement = document.querySelector(`.tab[data-file="${data.oldPath}"]`);
                if (tabElement) {
                    tabElement.dataset.file = data.newPath;
                    const span = tabElement.querySelector('span');
                    if (span) {
                        span.textContent = data.newName;
                    }
                }

                // Update current file if it was renamed
                if (this.currentFile === data.oldPath) {
                    this.currentFile = data.newPath;
                    document.getElementById('file-path').textContent = data.newName;
                }
            }
        });

        window.electronAPI.onFileDeleted((data) => {
            console.log('File deleted:', data);
            this.showNotification(`Deleted: ${data.filePath}`, 'success');
            this.refreshFileTree();

            // Close tab if it exists
            this.closeTab(data.filePath);
        });

        window.electronAPI.onFileChanged((data) => {
            console.log('File changed:', data);
            this.refreshFileTree();
        });

        // Menu actions
        window.electronAPI.onMenuAction((action) => {
            console.log('Menu action received:', action);
            this.handleMenuAction(action);
        });

        window.electronAPI.onAIAction((action) => {
            console.log('AI action received:', action);
            this.handleAIAction(action);
        });

        window.electronAPI.onWindowAction((action) => {
            console.log('Window action received:', action);
            this.handleWindowAction(action);
        });

        window.electronAPI.onTerminalAction((action) => {
            console.log('Terminal action received:', action);
            this.handleTerminalAction(action);
        });

        // Custom file change event
        window.electronAPI.onFileChangedExternally((data) => {
            this.handleExternalFileChange(data.filePath);
        });
    }

    async checkBackendHealth() {
        try {
            const health = await window.electronAPI.checkBackendHealth();
            this.updateBackendStatus(health);
        } catch (error) {
            console.error('Error checking backend health:', error);
            this.updateBackendStatus({
                success: false,
                status: 'error',
                message: 'Failed to check backend'
            });
        }
    }

    updateBackendStatus(health) {
        const statusElement = document.getElementById('backend-status');
        if (!statusElement) return;

        statusElement.style.display = 'block';
        const indicator = statusElement.querySelector('.status-indicator');
        const text = statusElement.querySelector('.status-text');

        if (health.success && health.status === 'healthy') {
            indicator.style.background = '#27ca3f';
            text.textContent = 'Backend: Connected';
            statusElement.style.background = '#2d2d2d';
            statusElement.classList.add('glass');
        } else {
            indicator.style.background = '#ff5f56';
            text.textContent = 'Backend: Disconnected';
            statusElement.style.background = '#ff5f5620';
            statusElement.classList.add('glass');

            // Show notification for first-time users
            if (!localStorage.getItem('backend-warning-shown')) {
                this.showNotification(
                    'AI Backend is not running. AI features will not work.\n\nStart the Flask backend:\n1. cd backend\n2. python app.py',
                    'warning'
                );
                localStorage.setItem('backend-warning-shown', 'true');
            }
        }
    }

    setupDefaultWorkspace() {
        console.log('Setting up default workspace...');
        this.fileTree = {
            name: 'Workspace',
            path: '/',
            type: 'folder',
            open: true,
            children: []
        };
        this.renderFileTree();
    }

    renderFileTree(node = this.fileTree, parentElement = document.getElementById('workspace'), depth = 0) {
        if (!parentElement) {
            console.error('Workspace element not found');
            return;
        }

        // Only clear if we are rendering the root
        if (depth === 0) {
            parentElement.innerHTML = '';

            // Add workspace info if we have a workspace
            if (this.workspacePath) {
                this.updateWorkspaceInfo(this.workspacePath);
            }
        }

        if (!node.children || node.children.length === 0) {
            if (depth === 0) {
                const emptyMsg = document.createElement('div');
                emptyMsg.className = 'empty-folder';
                emptyMsg.innerHTML = `
                    <i class="fas fa-folder-open" style="font-size: 48px; margin-bottom: 10px; opacity: 0.3;"></i>
                    <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 15px;">No files or folders</div>
                    <div style="display: flex; gap: 10px; justify-content: center;">
                        <button class="btn-modern" style="padding: 6px 12px; font-size: 11px;" 
                                onclick="codeEditor.openFileDialog()">
                            <i class="fas fa-file"></i> Open File
                        </button>
                        <button class="btn-modern" style="padding: 6px 12px; font-size: 11px;" 
                                onclick="codeEditor.openFolderDialog()">
                            <i class="fas fa-folder-plus"></i> Open Folder
                        </button>
                    </div>
                `;
                parentElement.appendChild(emptyMsg);
            }
            return;
        }

        // Sort: folders first, then files
        const sortedChildren = [...node.children].sort((a, b) => {
            if (a.type === 'folder' && b.type !== 'folder') return -1;
            if (a.type !== 'folder' && b.type === 'folder') return 1;
            return a.name.localeCompare(b.name);
        });

        const frag = document.createDocumentFragment();

        sortedChildren.forEach(child => {
            const element = document.createElement('div');
            // Use common class 'file-tree-row'? No, keeping existing classes but styled differently in CSS
            element.className = child.type === 'folder' ? 'folder-item' : 'file-item';
            element.dataset.path = child.path;

            // Highlight active file
            if (child.type === 'file' && this.currentFile === child.path) {
                element.classList.add('active');
            }

            const icon = child.type === 'folder' ?
                (child.open ? 'fa-folder-open' : 'fa-folder') :
                this.getFileIcon(child.name);

            // Indentation
            const paddingLeft = 10 + (depth * 15);

            element.innerHTML = `
                <div class="file-item-content" style="padding-left: ${paddingLeft}px;">
                    ${child.type === 'folder' ?
                    `<i class="fas fa-chevron-right folder-chevron" style="transform: ${child.open ? 'rotate(90deg)' : 'rotate(0deg)'}; transition: transform 0.2s;"></i>`
                    : '<span style="width:16px; margin-right:2px; display:inline-block;"></span>'}
                    
                    <i class="fas ${icon}" style="margin-right: 6px; width: 16px; text-align: center; color: ${child.type === 'folder' ? '#dcb67a' : 'inherit'}"></i>
                    <span class="file-name">${child.name}</span>
                </div>
                <div class="quick-file-actions">
                    <div class="quick-action-btn" onclick="event.stopPropagation(); codeEditor.showFileContextMenu('${child.path}', event)">
                        <i class="fas fa-ellipsis-h"></i>
                    </div>
                </div>
            `;

            const itemContainer = document.createElement('div');
            itemContainer.appendChild(element);

            // Append container to fragment first
            frag.appendChild(itemContainer);

            // Handle Folder Click
            element.addEventListener('click', (e) => {
                if (e.target.closest('.quick-file-actions')) return;

                if (child.type === 'folder') {
                    child.open = !child.open;
                    this.renderFileTree(); // Full re-render is simplest for consistency
                } else {
                    // Check if file is virtual (starts with /) or has stored content
                    if (child.path.startsWith('/')) {
                        // Virtual file - use stored content from tabs or empty
                        const tabInfo = this.tabs.get(child.path);
                        const content = tabInfo ? tabInfo.content : '';
                        this.openFile(child.path, content, child.name);
                    } else if (child.content !== undefined) {
                        // Has inline content
                        this.openFile(child.path, child.content, child.name);
                    } else if (this.tabs.has(child.path)) {
                        // Already open in a tab - use stored content
                        const tabInfo = this.tabs.get(child.path);
                        this.openFile(child.path, tabInfo.content || '', child.name);
                    } else {
                        // Normal file read
                        window.electronAPI.readFile(child.path).then(result => {
                            if (result.success) {
                                this.openFile(child.path, result.content, child.name);
                            } else {
                                this.showNotification(`Error opening file: ${result.error}`, 'error');
                            }
                        });
                    }

                    // Highlight active
                    document.querySelectorAll('.file-item').forEach(el => el.classList.remove('active'));
                    element.classList.add('active');
                }
            });

            // Recursive Children
            if (child.type === 'folder' && child.open) {
                // We create a container for children to keep DOM organized, though flattened is also fine.
                // But wait, if we use full re-render, we just need to append them here?
                // NO, we need to append them to itemContainer or frag?
                // Actually, renderFileTree clears the parent. So if we pass itemContainer as parent, it works.
                // BUT, my logic above says "Only clear if depth === 0". 
                // So:
                this.renderFileTree(child, itemContainer, depth + 1);
            }
        });

        parentElement.appendChild(frag);
    }

    getFileIcon(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const iconMap = {
            'js': 'fa-js-square',
            'jsx': 'fa-react',
            'ts': 'fa-js-square',
            'tsx': 'fa-react',
            'html': 'fa-html5',
            'htm': 'fa-html5',
            'css': 'fa-css3-alt',
            'scss': 'fa-sass',
            'sass': 'fa-sass',
            'less': 'fa-less',
            'json': 'fa-file-code',
            'md': 'fa-markdown',
            'py': 'fa-python',
            'java': 'fa-java',
            'c': 'fa-file-code',
            'cpp': 'fa-file-code',
            'h': 'fa-file-code',
            'hpp': 'fa-file-code',
            'cs': 'fa-microsoft',
            'php': 'fa-php',
            'rb': 'fa-gem',
            'go': 'fa-google',
            'rs': 'fa-rust',
            'sql': 'fa-database',
            'xml': 'fa-file-code',
            'yml': 'fa-file-code',
            'yaml': 'fa-file-code',
            'txt': 'fa-file-alt',
            'pdf': 'fa-file-pdf',
            'zip': 'fa-file-archive',
            'png': 'fa-file-image',
            'jpg': 'fa-file-image',
            'jpeg': 'fa-file-image',
            'gif': 'fa-file-image',
            'svg': 'fa-file-image',
            'vue': 'fa-vuejs',
            'svelte': 'fa-code'
        };

        return iconMap[ext] || 'fa-file';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    async openFile(filePath, content, fileName) {
        // Normalize path for consistent tab keys
        const originalPath = filePath;
        filePath = this.normalizePath(filePath);

        console.log(`Opening file: ${fileName} (${filePath})`);

        // IMPORTANT: Save current tab content BEFORE switching
        // Only save if we are actually SWITCHING tabs
        if (this.activeTab && this.activeTab !== filePath && this.tabs.has(this.activeTab) && this.editor) {
            const currentTabInfo = this.tabs.get(this.activeTab);
            currentTabInfo.content = this.editor.getValue();
            console.log(`Saved content for current tab: ${this.activeTab}`);
        }

        // Stop watching previous file (if switching)
        if (this.currentFile && this.currentFile !== filePath && this.fileSystemWatchers.has(this.currentFile)) {
            // Optional: Keep watchers alive for open tabs? 
            // Current logic closes watcher on switch? That's probably efficient but means background tabs don't update.
            // For now, keep existing behavior but use normalized check
            this.fileSystemWatchers.get(this.currentFile).close();
            this.fileSystemWatchers.delete(this.currentFile);
        }

        this.currentFile = filePath;

        // Add to explorer pane using ORIGINAL path for display compatibility if needed, 
        // but normalized for verification
        if (originalPath) {
            this.addToExplorer(originalPath, fileName || this.getFileNameFromPath(originalPath));
        }

        // Update or create tab
        if (!this.tabs.has(filePath)) {
            this.createTab(filePath, fileName);
        }

        // Store the content for this file
        const tabInfo = this.tabs.get(filePath);
        if (tabInfo) {
            // Only update content if provided (don't overwrite with empty if undefined)
            if (content !== undefined) {
                tabInfo.content = content;
            } else if (tabInfo.content === undefined) {
                tabInfo.content = '';
            }
            console.log(`Stored content for file: ${filePath}, length: ${tabInfo.content.length}`);
        }

        // Now activate the tab
        await this.setActiveTab(filePath);

        // Start watching the file using ORIGINAL path (fs.watch needs real path)
        if (originalPath && !originalPath.startsWith('/')) {
            this.watchFile(originalPath);
        }

        // Update UI
        document.getElementById('file-path').textContent = this.getFileNameFromPath(filePath) || fileName;
        document.getElementById('file-path').title = filePath;

        const languageStatus = document.getElementById('language-status');
        if (languageStatus) {
            const language = this.getLanguageFromExtension(fileName);
            languageStatus.querySelector('span').textContent =
                language.charAt(0).toUpperCase() + language.slice(1);
        }

        this.updateOutline();
        this.updateFileInfo(filePath);
    }

    async openFileFromPath(filePath, fileName) {
        try {
            const content = await window.electronAPI.readFileContent(filePath);
            this.openFile(filePath, content, fileName);
        } catch (error) {
            console.error('Error opening file:', error);
            this.showNotification(`Error opening file: ${error.message}`, 'error');
        }
    }

    watchFile(filePath) {
        try {
            const fs = require('fs');
            const watcher = fs.watch(filePath, (eventType, filename) => {
                if (eventType === 'change' && filename) {
                    console.log(`File changed externally: ${filePath}`);
                    this.handleExternalFileChange(filePath);
                }
            });

            this.fileSystemWatchers.set(filePath, watcher);
            console.log(`Started watching file: ${filePath}`);
        } catch (error) {
            console.error('Error setting up file watcher:', error);
        }
    }

    handleExternalFileChange(filePath) {
        // Check if we're the ones who made the change
        if (this.currentFile === filePath && this.editor) {
            const currentContent = this.editor.getValue();

            // Read file content
            try {
                const fs = require('fs');
                const fileContent = fs.readFileSync(filePath, 'utf-8');

                // Only show notification if content is different
                if (currentContent !== fileContent) {
                    this.showNotification('File changed externally. Reloading...', 'warning');

                    // Update editor content
                    setTimeout(() => {
                        if (this.editor) {
                            const model = this.editor.getModel();
                            if (model) {
                                model.setValue(fileContent);
                            }
                        }
                        this.showNotification('File reloaded from disk', 'success');
                    }, 100);
                }
            } catch (error) {
                console.error('Error reading changed file:', error);
            }
        }
    }

    createTab(filePath, fileName) {
        console.log(`Creating tab for: ${fileName}`);

        const tabsContainer = document.getElementById('tabs-container');
        if (!tabsContainer) {
            console.error('Tabs container not found');
            return;
        }

        const tab = document.createElement('div');
        tab.className = 'tab hover-lift';
        tab.dataset.file = filePath;

        const icon = this.getFileIcon(fileName);
        tab.innerHTML = `
            <i class="fas ${icon}"></i>
            <span>${fileName}</span>
            <i class="fas fa-times tab-close"></i>
        `;

        tabsContainer.appendChild(tab);
        this.tabs.set(filePath, { element: tab, fileName, saved: true });

        console.log(`Tab created: ${fileName}`);
    }

    async setActiveTab(filePath) {
        // Normalize
        filePath = this.normalizePath(filePath);
        console.log(`Setting active tab: ${filePath}`);

        // Save content of current tab ONLY if switching to a DIFFERENT tab
        // This prevents overwriting updated content when refreshing the same tab (e.g. after Apply Edit)
        if (this.activeTab && this.activeTab !== filePath && this.tabs.has(this.activeTab) && this.editor) {
            const currentTabInfo = this.tabs.get(this.activeTab);
            currentTabInfo.content = this.editor.getValue();
            console.log(`Saved content for tab: ${this.activeTab}`);
        }

        // Remove active class from all tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.remove('active');
            tab.classList.remove('neon-glow');
        });

        if (this.tabs.has(filePath)) {
            const tabInfo = this.tabs.get(filePath);
            const tab = tabInfo.element;
            tab.classList.add('active');
            tab.classList.add('neon-glow');
            this.activeTab = filePath;
            this.currentFile = filePath;

            // Handle Diff Tabs
            if (tabInfo.isDiff && tabInfo.editData) {
                console.log('Activating Diff Tab for:', filePath);
                // Ensure Diff container exists
                let diffContainer = document.getElementById('diff-editor-container-tab');
                if (!diffContainer) {
                    diffContainer = document.createElement('div');
                    diffContainer.id = 'diff-editor-container-tab';
                    diffContainer.style.position = 'absolute';
                    diffContainer.style.top = '0';
                    diffContainer.style.left = '0';
                    diffContainer.style.width = '100%';
                    diffContainer.style.height = '100%';
                    diffContainer.style.zIndex = '9999'; // Very high z-index
                    diffContainer.style.backgroundColor = '#1e1e1e'; // Opaque background
                    document.getElementById('editor').appendChild(diffContainer);

                    this.diffEditorInstance = monaco.editor.createDiffEditor(diffContainer, {
                        theme: 'vs-dark',
                        automaticLayout: true,
                        readOnly: true,
                        originalEditable: false,
                        renderSideBySide: false // Inline Diff
                    });
                }
                diffContainer.style.display = 'block';

                const edit = tabInfo.editData;
                const originalModel = monaco.editor.createModel(edit.original_content, this.getLanguageFromExtension(edit.file_path));
                const modifiedModel = monaco.editor.createModel(edit.new_content, this.getLanguageFromExtension(edit.file_path));

                this.diffEditorInstance.setModel({
                    original: originalModel,
                    modified: modifiedModel
                });

                // Force layout update
                setTimeout(() => {
                    this.diffEditorInstance.layout();
                }, 50);

                this.renderDiffActions(diffContainer, tabInfo.editData, filePath);

            } else {
                // Standard Editor Tab
                // Hide diff container if exists
                const diffContainer = document.getElementById('diff-editor-container-tab');
                if (diffContainer) {
                    diffContainer.style.display = 'none';
                }

                // Update editor content from stored content
                if (this.editor) {
                    const contentToSet = tabInfo.content !== undefined ? tabInfo.content : '';

                    // Only setValue if it's different to prevent cursor jumping or unnecessary updates
                    // But if we are refreshing after edit, we MUST update.
                    // editor.getValue() might be stale if we just hid diff view.

                    if (this.editor.getValue() !== contentToSet) {
                        this.editor.setValue(contentToSet);
                    }
                    console.log(`Loaded content for tab: ${filePath}, length: ${contentToSet.length}`);

                    // Update language
                    const fileName = tabInfo.fileName;
                    const language = this.getLanguageFromExtension(fileName);
                    const model = this.editor.getModel();
                    if (model) {
                        monaco.editor.setModelLanguage(model, language);
                    }
                }

                // Update UI elements for standard tabs
                const fileName = tabInfo.fileName;
                document.getElementById('file-path').textContent = this.getFileNameFromPath(filePath) || fileName;
                document.getElementById('file-path').title = filePath;

                const languageStatus = document.getElementById('language-status');
                if (languageStatus) {
                    const language = this.getLanguageFromExtension(fileName);
                    languageStatus.querySelector('span').textContent =
                        language.charAt(0).toUpperCase() + language.slice(1);
                }

                this.updateOutline();
            }
        }

        // Update active file in sidebar
        document.querySelectorAll('.file-item, .folder-item').forEach(item => {
            item.classList.remove('active');
            // Normalize sidebar path to ensure match
            const itemPath = this.normalizePath(item.dataset.path);
            if (itemPath === filePath) {
                item.classList.add('active');
            }
        });

        console.log(`Active tab set: ${filePath}`);
    }

    renderDiffActions(container, edit, filePath) {
        let toolbar = document.getElementById('diff-actions-toolbar');
        if (!toolbar) {
            toolbar = document.createElement('div');
            toolbar.id = 'diff-actions-toolbar';
            toolbar.style.cssText = `
                position: absolute;
                top: 10px;
                right: 20px;
                z-index: 10000;
                display: flex;
                gap: 10px;
                background: rgba(30, 30, 30, 0.8);
                backdrop-filter: blur(4px);
                padding: 5px;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            `;
            container.appendChild(toolbar);
        }

        toolbar.innerHTML = '';

        const acceptBtn = document.createElement('button');
        acceptBtn.className = 'diff-overlay-btn btn-accept-change';
        acceptBtn.innerHTML = '<i class="fas fa-check"></i> Accept';
        acceptBtn.onclick = () => this.applySingleEdit(edit);

        const rejectBtn = document.createElement('button');
        rejectBtn.className = 'diff-overlay-btn btn-reject-change';
        rejectBtn.innerHTML = '<i class="fas fa-times"></i> Reject';
        rejectBtn.onclick = () => this.rejectSingleEdit(edit);

        toolbar.appendChild(acceptBtn);
        toolbar.appendChild(rejectBtn);
    }

    normalizePath(filePath) {
        if (!filePath) return '';
        // Normalize slashes and casing for consistent Map keys
        // Use backend logic: lower case drive letters, standard slashes
        let normalized = filePath.replace(/\\/g, '/');
        // If on Windows (heuristic), lowercase valid drive letter
        if (normalized.match(/^[a-zA-Z]:\//)) {
            normalized = normalized.charAt(0).toLowerCase() + normalized.slice(1);
        }
        return normalized; // Simple normalization for now. 
        // ideally we'd use path.resolve() but that's node-specific and we want consistent string keys
    }

    async applySingleEdit(edit) {
        try {
            // Normalize properly
            let editPath = this.normalizePath(edit.file_path);

            // Determine effective file path
            let targetPath = edit.file_path; // Keep original for FS ops

            const isNew = edit.is_new;
            if (isNew) {
                const fileName = this.getFileNameFromPath(edit.file_path);

                targetPath = await window.electronAPI.createFile({
                    folderPath: this.workspacePath,
                    fileName: fileName
                });
                console.log(`Created new file at: ${targetPath}`);
            }

            await window.electronAPI.saveFile({
                filePath: targetPath,
                content: edit.new_content
            });

            // Update Tab State
            // Try matching normalized
            let tabFilePath = editPath;
            if (!this.tabs.has(tabFilePath)) {
                // Try finding by fuzzy match or original?
                // If not found, check if we have it under original
                if (this.tabs.has(edit.file_path)) tabFilePath = edit.file_path;
            }

            if (this.tabs.has(tabFilePath)) {
                const tabInfo = this.tabs.get(tabFilePath);

                // CRITICAL SEQUENCE:
                // 1. Update state data first
                tabInfo.isDiff = false;
                tabInfo.editData = null;
                tabInfo.content = edit.new_content;

                // 2. If this is the active tab, directly update editor to prevent setActiveTab from overwriting with stale data
                if (this.activeTab === tabFilePath) {
                    if (this.editor) {
                        this.editor.setValue(edit.new_content);
                    }

                    // Hide Diff View Manually
                    const diffContainer = document.getElementById('diff-editor-container-tab');
                    if (diffContainer) diffContainer.style.display = 'none';

                    // Remove review mode class
                    tabInfo.element.classList.remove('review-mode');
                }

                // 3. Force UI Refresh without saving stale content
                // We pass a flag or just call it, but we need to ensure setActiveTab doesn't save *this* tab's stale content
                // Refactor setActiveTab to check `this.activeTab !== filePath` for saving.
                await this.setActiveTab(tabFilePath);
            }
            this.showNotification(`Changes applied to ${this.getFileNameFromPath(targetPath)}`, 'success');

            // Refresh file tree
            this.refreshFileTree();

        } catch (error) {
            this.showNotification(`Error applying edit: ${error.message}`, 'error');
        }
    }

    rejectSingleEdit(edit) {
        const filePath = edit.file_path;
        if (this.tabs.has(filePath)) {
            const tabInfo = this.tabs.get(filePath);

            if (edit.is_new) {
                // If it was a new file, just close the tab
                this.closeTab(filePath);
            } else {
                // If existing file, revert to normal view
                tabInfo.isDiff = false;
                tabInfo.editData = null;
                tabInfo.content = edit.original_content;
                this.setActiveTab(filePath);
            }
            this.showNotification('Edit rejected', 'info');
        }
    }

    closeTab(filePath) {
        filePath = this.normalizePath(filePath);
        console.log(`Closing tab: ${filePath}`);

        // Stop watching file if no longer open
        // We use normalized keys now for watchers too in logic? 
        // Logic in openFile: fileSystemWatchers.delete(this.currentFile) -> currentFile is normalized.
        // So yes.
        if (this.fileSystemWatchers.has(filePath)) {
            this.fileSystemWatchers.get(filePath).close();
            this.fileSystemWatchers.delete(filePath);
        }

        if (this.tabs.has(filePath)) {
            const tab = this.tabs.get(filePath).element;
            tab.remove();
            this.tabs.delete(filePath);

            if (this.activeTab === filePath) {
                const remainingTabs = Array.from(this.tabs.keys());
                if (remainingTabs.length > 0) {
                    this.setActiveTab(remainingTabs[0]);
                } else {
                    // No tabs left - Clear editor
                    this.currentFile = null;
                    this.activeTab = null;
                    if (this.editor) {
                        this.editor.setValue('');
                        document.getElementById('file-path').textContent = '';
                        document.getElementById('language-status').querySelector('span').textContent = 'Plain Text';
                    }
                }
            }
        }
    }

    closeAllTabs() {
        console.log('Closing all tabs');

        // Stop all file watchers
        this.fileSystemWatchers.forEach(watcher => watcher.close());
        this.fileSystemWatchers.clear();

        this.tabs.clear();
        const tabsContainer = document.getElementById('tabs-container');
        if (tabsContainer) {
            tabsContainer.innerHTML = '';
        }

        // Reset Editor State
        this.currentFile = null;
        this.activeTab = null;
        if (this.editor) {
            this.editor.setValue('');
            document.getElementById('file-path').textContent = '';
            const langStatus = document.getElementById('language-status');
            if (langStatus) langStatus.querySelector('span').textContent = 'Plain Text';
        }
    }

    createNewFile() {
        this.showNewFileDialog();
    }

    showNewFileDialog() {
        const dialog = document.getElementById('new-file-dialog');
        const input = document.getElementById('new-file-input');
        if (dialog && input) {
            input.value = 'newfile.js';
            dialog.style.display = 'flex';
            input.focus();
            input.select();
        }
    }

    hideNewFileDialog() {
        const dialog = document.getElementById('new-file-dialog');
        if (dialog) {
            dialog.style.display = 'none';
        }
    }

    confirmNewFile() {
        const input = document.getElementById('new-file-input');
        const fileName = input ? input.value.trim() : '';

        if (!fileName) {
            this.showNotification('Please enter a file name', 'warning');
            return;
        }

        this.hideNewFileDialog();

        if (this.workspacePath) {
            window.electronAPI.createFile({
                folderPath: this.workspacePath,
                fileName: fileName
            }).then(filePath => {
                this.openFile(filePath, '', fileName);
            }).catch(error => {
                this.showNotification(`Failed to create file: ${error.message}`, 'error');
            });
        } else {
            // Create an untitled file that opens immediately
            // Use a virtual path prefix to indicate it's not saved yet
            const virtualPath = `/untitled/${fileName}`;
            this.openFile(virtualPath, '', fileName);
            this.showNotification('File created. Use Save As to choose a location.', 'info');
        }
    }

    createNewFolder() {
        this.showNewFolderDialog();
    }

    showNewFolderDialog() {
        const dialog = document.getElementById('new-folder-dialog');
        const input = document.getElementById('new-folder-input');
        if (dialog && input) {
            input.value = 'new_folder';
            dialog.style.display = 'flex';
            input.focus();
            input.select();
        }
    }

    hideNewFolderDialog() {
        const dialog = document.getElementById('new-folder-dialog');
        if (dialog) {
            dialog.style.display = 'none';
        }
    }

    async confirmNewFolder() {
        const input = document.getElementById('new-folder-input');
        const folderName = input ? input.value.trim() : '';

        if (!folderName) {
            this.showNotification('Please enter a folder name', 'warning');
            return;
        }

        this.hideNewFolderDialog();

        try {
            await window.electronAPI.createFolder({
                folderPath: this.workspacePath || null,
                folderName: folderName
            });
            this.showNotification(`Folder "${folderName}" created successfully`, 'success');
        } catch (error) {
            if (error && error.message) {
                this.showNotification(`Failed to create folder: ${error.message}`, 'error');
            }
            // User may have cancelled the dialog, which is fine
        }
    }

    showRenameDialog(filePath, currentName) {
        const dialog = document.getElementById('rename-dialog');
        const input = document.getElementById('rename-input');

        this.renameContext = { filePath, currentName };
        input.value = currentName;
        dialog.style.display = 'flex';
        input.focus();
        input.select();
    }

    hideRenameDialog() {
        const dialog = document.getElementById('rename-dialog');
        dialog.style.display = 'none';
        this.renameContext = null;
    }

    async confirmRename() {
        if (!this.renameContext) return;

        const { filePath, currentName } = this.renameContext;
        const newName = document.getElementById('rename-input').value.trim();

        if (!newName || newName === currentName) {
            this.hideRenameDialog();
            return;
        }

        try {
            await window.electronAPI.renameFile({
                oldPath: filePath,
                newName: newName
            });
        } catch (error) {
            this.showNotification(`Failed to rename: ${error.message}`, 'error');
        }

        this.hideRenameDialog();
    }

    async saveCurrentFile() {
        if (!this.editor) return;

        const content = this.editor.getValue();
        const filePath = this.currentFile;

        if (!filePath || filePath.startsWith('/new/') || filePath.startsWith('/default/') || filePath.startsWith('/welcome/') || filePath.startsWith('/untitled/')) {
            await this.saveFileAs();
            return;
        }

        try {
            await window.electronAPI.saveFile({
                filePath: filePath,
                content: content
            });

            // Update tab status
            if (this.tabs.has(filePath)) {
                const tab = this.tabs.get(filePath);
                tab.saved = true;
                tab.element.classList.remove('unsaved');
            }

            this.showNotification('File saved successfully', 'success');
        } catch (error) {
            this.showNotification(`Failed to save: ${error.message}`, 'error');
        }
    }

    async saveFileAs() {
        if (!this.editor) return;

        const content = this.editor.getValue();
        const currentFileName = this.currentFile ? this.getFileNameFromPath(this.currentFile) : 'untitled.js';

        this.showSaveAsDialog(content, currentFileName);
    }

    showSaveAsDialog(content, suggestedName) {
        const input = document.createElement('input');
        input.type = 'file';
        input.nwsaveas = suggestedName;
        input.accept = '.js,.ts,.jsx,.tsx,.html,.css,.py,.json,.md,.txt,.xml,.yaml,.yml,.java,.c,.cpp,.h,.cs,.php,.rb,.go,.rs,.sql';

        input.onchange = async (e) => {
            const filePath = e.target.files[0]?.path;
            if (filePath) {
                try {
                    await window.electronAPI.saveFile({
                        filePath: filePath,
                        content: content
                    });

                    const fileName = this.getFileNameFromPath(filePath);

                    // Close the old untitled tab if it exists
                    const oldPath = this.currentFile;
                    if (oldPath && oldPath.startsWith('/untitled/')) {
                        this.closeTab(oldPath);
                    }

                    this.openFile(filePath, content, fileName);
                    this.showNotification('File saved successfully', 'success');
                } catch (error) {
                    this.showNotification(`Failed to save: ${error.message}`, 'error');
                }
            }
        };

        input.click();
    }

    async deleteFile(filePath) {
        if (!filePath || filePath.startsWith('/')) {
            this.showNotification('Cannot delete default files', 'warning');
            return;
        }

        if (confirm('Are you sure you want to delete this file?')) {
            try {
                await window.electronAPI.deleteFile({ filePath });
            } catch (error) {
                this.showNotification(`Failed to delete: ${error.message}`, 'error');
            }
        }
    }

    async openFolder(folderPath, files, folderStructure) {
        console.log('Opening folder:', folderPath);

        this.workspacePath = folderPath;

        // Reset Workspace State
        this.closeAllTabs();
        this.clearAIChat();

        // Reset Chat and RAG
        const chatMessages = document.getElementById('ai-chat-messages');
        if (chatMessages) chatMessages.innerHTML = '';
        this.aiChatHistory = [];

        const ragToggle = document.getElementById('rag-toggle');
        if (ragToggle) ragToggle.checked = false;

        window.electronAPI.resetRAGIndex().catch(err => console.error('Failed to reset RAG:', err));

        // Use the folder structure if provided
        if (folderStructure) {
            this.fileTree = folderStructure;
        } else {
            // Build file tree from files
            this.fileTree = {
                name: this.getFileNameFromPath(folderPath),
                path: folderPath,
                type: 'folder',
                open: true,
                children: files || []
            };
        }

        this.renderFileTree();
        this.updateWorkspaceInfo(folderPath);

        // Start watching folder
        this.watchWorkspaceFolder(folderPath);

        this.showNotification(`Workspace opened: ${folderPath}`, 'success');
    }

    async openFolderFromPath(folderPath) {
        try {
            const structure = await window.electronAPI.getFolderStructure(folderPath);
            this.openFolder(folderPath, structure, {
                name: this.getFileNameFromPath(folderPath),
                path: folderPath,
                type: 'folder',
                open: true,
                children: structure
            });
        } catch (error) {
            console.error('Error opening folder:', error);
            this.showNotification(`Error opening folder: ${error.message}`, 'error');
        }
    }

    watchWorkspaceFolder(folderPath) {
        try {
            const fs = require('fs');
            const path = require('path');

            const watcher = fs.watch(folderPath, { recursive: true }, (eventType, filename) => {
                if (filename) {
                    const filePath = path.join(folderPath, filename);
                    console.log(`File system event: ${eventType} ${filename}`);

                    setTimeout(() => {
                        this.refreshFileTree();
                    }, 100);

                    if (eventType === 'change' && this.currentFile === filePath) {
                        this.handleExternalFileChange(filePath);
                    }
                }
            });

            this.fileWatchers.set(folderPath, watcher);
        } catch (error) {
            console.error('Error watching folder:', error);
        }
    }

    async refreshFileTree() {
        if (this.workspacePath) {
            try {
                const structure = await window.electronAPI.getFolderStructure(this.workspacePath);
                this.fileTree = {
                    name: this.getFileNameFromPath(this.workspacePath),
                    path: this.workspacePath,
                    type: 'folder',
                    open: true,
                    children: structure
                };
                this.renderFileTree();
            } catch (error) {
                console.error('Error refreshing file tree:', error);
            }
        }
    }

    async runCurrentFile() {
        if (!this.currentFile) {
            this.showNotification('No file to run', 'warning');
            return;
        }

        const language = this.getLanguageFromExtension(this.currentFile);

        this.showTerminal();
        this.addTerminalOutput(`$ Running ${this.currentFile}...`);

        try {
            const result = await window.electronAPI.runCode({
                filePath: this.currentFile,
                language: language
            });

            if (result.success) {
                if (result.output.stdout) {
                    this.addTerminalOutput(result.output.stdout);
                } else {
                    this.addTerminalOutput('Program executed successfully');
                }
            } else {
                this.addTerminalOutput(`Error: ${result.error}`, 'error');
            }
        } catch (error) {
            this.addTerminalOutput(`Execution error: ${error.message}`, 'error');
        }
    }

    toggleAIPanel() {
        const aiPanel = document.getElementById('ai-panel');
        if (!aiPanel) {
            console.error('AI panel element not found');
            return;
        }

        this.aiPanelVisible = !this.aiPanelVisible;

        if (this.aiPanelVisible) {
            aiPanel.classList.remove('collapsed');
        } else {
            aiPanel.classList.add('collapsed');
        }
    }

    toggleTerminal() {
        const terminalPanel = document.getElementById('terminal-panel');
        if (!terminalPanel) {
            console.error('Terminal panel element not found');
            return;
        }

        this.terminalVisible = !this.terminalVisible;

        if (this.terminalVisible) {
            terminalPanel.classList.add('visible');
        } else {
            terminalPanel.classList.remove('visible');
        }
    }

    showTerminal() {
        this.terminalVisible = true;
        const terminalPanel = document.getElementById('terminal-panel');
        if (terminalPanel) {
            terminalPanel.classList.add('visible');
        }
    }

    clearTerminal() {
        const terminalOutput = document.querySelector('.terminal-output');
        if (terminalOutput) {
            terminalOutput.innerHTML = '';
        }
    }

    newTerminal() {
        this.showTerminal();
        this.addTerminalOutput('$ New terminal created');
    }

    addTerminalOutput(output, type = 'normal') {
        const terminalContent = document.getElementById('terminal-content');
        if (!terminalContent) {
            console.error('Terminal content element not found');
            return;
        }

        const outputDiv = document.createElement('div');
        outputDiv.className = `terminal-line ${type}`;
        outputDiv.textContent = output;

        const terminalOutput = terminalContent.querySelector('.terminal-output');
        if (terminalOutput) {
            terminalOutput.appendChild(outputDiv);
            terminalContent.scrollTop = terminalContent.scrollHeight;
        }
    }

    executeTerminalCommand(command) {
        this.addTerminalOutput(`$ ${command}`);

        switch (command.trim().toLowerCase()) {
            case 'help':
                this.addTerminalOutput('Available commands:');
                this.addTerminalOutput('  help - Show this help message');
                this.addTerminalOutput('  clear - Clear terminal');
                this.addTerminalOutput('  ls - List files in workspace');
                this.addTerminalOutput('  run - Run current file');
                this.addTerminalOutput('  date - Show current date and time');
                this.addTerminalOutput('  echo [text] - Echo text');
                this.addTerminalOutput('  backend - Check backend status');
                this.addTerminalOutput('  open [file] - Open a file');
                this.addTerminalOutput('  pwd - Show current directory');
                break;
            case 'clear':
                const terminalOutput = document.querySelector('.terminal-output');
                if (terminalOutput) {
                    terminalOutput.innerHTML = '';
                }
                break;
            case 'ls':
                this.addTerminalOutput('Files in workspace:');
                if (this.fileTree.children) {
                    this.fileTree.children.forEach(child => {
                        this.addTerminalOutput(`  ${child.name} (${child.type})`);
                    });
                } else {
                    this.addTerminalOutput('  No workspace open');
                }
                break;
            case 'run':
                this.runCurrentFile();
                break;
            case 'date':
                this.addTerminalOutput(new Date().toString());
                break;
            case 'backend':
                this.checkBackendHealth();
                this.addTerminalOutput('Checking backend status...');
                break;
            case 'pwd':
                this.addTerminalOutput(this.workspacePath || 'No workspace open');
                break;
            default:
                if (command.startsWith('echo ')) {
                    this.addTerminalOutput(command.substring(5));
                } else if (command.startsWith('open ')) {
                    const fileName = command.substring(5).trim();
                    this.openFileFromTerminal(fileName);
                } else {
                    this.addTerminalOutput(`Command not found: ${command}`, 'error');
                }
        }
    }

    openFileFromTerminal(fileName) {
        if (!this.workspacePath) {
            this.addTerminalOutput('Error: No workspace open', 'error');
            return;
        }

        const path = require('path');
        const filePath = path.join(this.workspacePath, fileName);

        try {
            const fs = require('fs');
            if (fs.existsSync(filePath)) {
                const content = fs.readFileSync(filePath, 'utf-8');
                this.openFile(filePath, content, fileName);
                this.addTerminalOutput(`Opened file: ${fileName}`);
            } else {
                this.addTerminalOutput(`File not found: ${fileName}`, 'error');
            }
        } catch (error) {
            this.addTerminalOutput(`Error opening file: ${error.message}`, 'error');
        }
    }

    async handleAIAction(action) {
        console.log(`Handling AI action: ${action}`);

        if (!this.editor) {
            this.showNotification('Editor not initialized', 'error');
            return;
        }

        const code = this.editor.getValue();
        const selection = this.editor.getSelection();
        const selectedText = selection ?
            this.editor.getModel().getValueInRange(selection) : '';

        const language = this.getLanguageFromExtension(this.currentFile || 'script.js');

        this.showNotification('Connecting to AI backend...', 'info');

        try {
            let result;

            switch (action) {
                case 'generate':
                    result = await window.electronAPI.generateCode(
                        selectedText || 'Generate a useful code snippet',
                        '',
                        language
                    );
                    if (result.success) {
                        this.editor.setValue(result.code);
                        this.showNotification('Code generated successfully', 'success');
                    } else {
                        this.showNotification(`AI Error: ${result.error}`, 'error');
                    }
                    break;

                case 'explain':
                    result = await window.electronAPI.explainCode(
                        selectedText || code,
                        language
                    );
                    if (result.success) {
                        this.addAIMessage('ai', `**Explanation:**\n\n${result.explanation}`);
                    } else {
                        this.showNotification(`Explain Error: ${result.error}`, 'error');
                    }
                    break;

                case 'debug':
                    result = await window.electronAPI.debugCode(
                        selectedText || code,
                        language
                    );
                    if (result.success) {
                        this.editor.setValue(result.debugged_code);
                        this.addAIMessage('ai', `**Debug Result:**\n\n${result.explanation}`);
                    } else {
                        this.showNotification(`Debug Error: ${result.error}`, 'error');
                    }
                    break;

                case 'optimize':
                    result = await window.electronAPI.optimizeCode(
                        selectedText || code,
                        language,
                        'performance'
                    );
                    if (result.success) {
                        this.editor.setValue(result.optimized_code);
                        this.addAIMessage('ai', `**Optimization Result:**\n\n${result.explanation}`);
                    } else {
                        this.showNotification(`Optimize Error: ${result.error}`, 'error');
                    }
                    break;

                case 'test':
                    result = await window.electronAPI.writeTests(
                        selectedText || code,
                        language
                    );
                    if (result.success) {
                        this.addAIMessage('ai', `**Tests Generated:**\n\n\`\`\`${language}\n${result.tests}\n\`\`\`\n\n${result.test_explanation}`);
                    } else {
                        this.showNotification(`Test Error: ${result.error}`, 'error');
                    }
                    break;

                case 'document':
                    result = await window.electronAPI.documentCode(
                        selectedText || code,
                        language
                    );
                    if (result.success) {
                        this.editor.setValue(result.documented_code);
                        this.addAIMessage('ai', `**Documentation added:**\n\n${result.documentation}`);
                    } else {
                        this.showNotification(`Document Error: ${result.error}`, 'error');
                    }
                    break;
                    break;
                case 'edit':
                    // Multi-file edit requested
                    // The backend should return a plan or list of edits
                    // We'll treat this differently - maybe call a different endpoint or handle 'task' param

                    // For now, let's assume specific endpoint usage if action is 'edit'
                    // Implementation below
                    break;
            }

        } catch (error) {
            console.error('AI action error:', error);
            this.showNotification(`AI Error: ${error.message}`, 'error');
        }
    }

    async handleMultiFileEdit(task) {
        console.log("HandleMultiFileEdit: Starting Native Diff Workflow");

        // create a placeholder message for progress updates
        const progressId = 'progress-' + Date.now();
        this.addAIMessage('ai', '<span id="' + progressId + '">Analyzing request...</span>');

        // Start progress simulation
        const stopProgress = this.simulateProgress(progressId);

        try {
            const files = Array.from(this.tabs.keys());

            const result = await window.electronAPI.invoke('agent-edit', {
                task: task,
                files: files
            });

            // Stop progress simulation
            stopProgress();

            if (result.success) {
                // Remove the progress message or update it to the result
                const progressElement = document.getElementById(progressId);
                if (progressElement) {
                    // Find the parent message div and remove it to avoid clutter, 
                    // OR update it. Let's remove it and add the real response.
                    const messageDiv = progressElement.closest('.message');
                    if (messageDiv) messageDiv.remove();
                }

                // If there are edits, show the full UI
                // If there are edits, show the full UI
                if (result.edits && result.edits.length > 0) {
                    this.addAIMessage('ai', `**Plan Created:**\n\n${result.plan}\n\nI have proposed edits for ${result.edits.length} files. Opening files for review...`);

                    // Auto-open diffs in native tabs
                    this.currentEdits = result.edits;
                    for (const edit of result.edits) {
                        await this.openEditAsDiff(edit);
                    }

                    // Add "Apply" action to chat
                    this.addChatAction('Apply All Changes', () => this.applyAllEdits(result.edits));
                } else {
                    // No edits, treat as a normal chat response (using the plan as the response)
                    // The backend typically returns the LLM response in 'plan' if no edits were found
                    this.addAIMessage('ai', result.plan || "I processed your request but found no changes needed.");
                }

            } else {
                const progressElement = document.getElementById(progressId);
                if (progressElement) {
                    const messageDiv = progressElement.closest('.message');
                    if (messageDiv) messageDiv.remove();
                }
                this.addAIMessage('ai', `**Error:** Failed to generate response: ${result.error}`);
            }
        } catch (error) {
            stopProgress();
            const progressElement = document.getElementById(progressId);
            if (progressElement) {
                const messageDiv = progressElement.closest('.message');
                if (messageDiv) messageDiv.remove();
            }
            this.showNotification(`Error: ${error.message}`, 'error');
            this.addAIMessage('ai', `**System Error:** ${error.message}`);
        }
    }

    simulateProgress(elementId) {
        const phases = [
            "Analyzing context...",
            "Reading files...",
            "Planning changes...",
            "Designing solution...",
            "Generating code...",
            "Reviewing changes...",
            "Finalizing..."
        ];

        let phaseIndex = 0;
        const element = document.getElementById(elementId);

        const interval = setInterval(() => {
            const el = document.getElementById(elementId);
            if (el) {
                phaseIndex = (phaseIndex + 1) % phases.length;
                el.textContent = phases[phaseIndex];
                // Optional: Scroll to bottom if needed
                const messages = document.getElementById('ai-chat-messages');
                if (messages) messages.scrollTop = messages.scrollHeight;
            }
        }, 2000); // Update every 2 seconds

        return () => clearInterval(interval);
    }

    async openEditAsDiff(edit) {
        const filePath = edit.file_path;

        // ensure file is open (creates tab if needed)
        if (edit.is_new) {
            const fileName = this.getFileNameFromPath(filePath);
            if (!this.tabs.has(filePath)) {
                this.createTab(filePath, fileName);
            }
        } else {
            // Open existing file
            try {
                // We use openFile to ensure tab exists and content loaded
                await this.openFile(filePath, null, this.getFileNameFromPath(filePath));
            } catch (e) {
                console.warn("Could not open file from disk (maybe new?):", e);
            }
        }

        // hijacked tab for diff view
        if (this.tabs.has(filePath)) {
            const tabInfo = this.tabs.get(filePath);
            tabInfo.isDiff = true;
            tabInfo.editData = edit;

            // Force refresh to show diff
            this.setActiveTab(filePath);
        }
    }

    // Override/Extend setTabContent to handle diff editors?
    // Current architecture creates one editor `this.editor`.
    // We need to support swapping the editor instance or model.
    // simpler approach: When active tab is a Diff tab, dispose default editor (or hide) and show diff editor.

    // NOTE: For this step, I'll modify setActiveTab in a subsequent edit or assume I can inject logic here. 
    // Actually, `setActiveTab` logic is complex. Let's try to hook into the existing system by:
    // 1. Modifying `setActiveTab` to check for `isDiff` flag.
    // 2. If diff, render DiffEditor.

    async applyAllEdits(edits) {
        this.showNotification('Applying all changes...', 'info');
        try {
            for (const edit of edits) {
                // Save to disk
                const isNew = edit.is_new;

                if (isNew) {
                    await window.electronAPI.createFile({
                        folderPath: this.workspacePath,
                        fileName: this.getFileNameFromPath(edit.file_path)
                    });
                }

                await window.electronAPI.saveFile({
                    filePath: edit.file_path,
                    content: edit.new_content
                });

                // Reset Tab State
                if (this.tabs.has(edit.file_path)) {
                    const tabInfo = this.tabs.get(edit.file_path);
                    tabInfo.isDiff = false;
                    tabInfo.editData = null;
                    tabInfo.content = edit.new_content; // Update stored content

                    // If this is the active tab, reload to show standard editor
                    if (this.activeTab === edit.file_path) {
                        this.setActiveTab(edit.file_path);
                    }
                }
            }
            this.showNotification('Changes applied!', 'success');
            this.refreshFileTree();

            // Reload current tab just in case
            if (this.activeTab) this.setActiveTab(this.activeTab);

        } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
        }
    }

    async sendAIMessage() {
        const input = document.getElementById('ai-input');
        if (!input) {
            console.error('AI input element not found');
            return;
        }

        const message = input.value.trim();

        if (!message) {
            this.showNotification('Please enter a message', 'warning');
            return;
        }

        // Detect if this is an explicit edit command
        if (message.toLowerCase().startsWith('/edit') || message.toLowerCase().includes('change current file')) {
            // For now, hook into chat, but ideally we parse intent
        }

        this.addAIMessage('user', message);

        input.value = '';

        this.showAITypingIndicator();

        try {
            const messages = document.getElementById('ai-chat-messages');
            const history = [];
            if (messages) {
                const messageElements = messages.querySelectorAll('.message');
                messageElements.forEach(el => {
                    if (el.classList.contains('user-message')) {
                        history.push({
                            role: 'user',
                            content: el.querySelector('.message-text').textContent
                        });
                    } else if (el.classList.contains('ai-message')) {
                        history.push({
                            role: 'assistant',
                            content: el.querySelector('.message-text').textContent
                        });
                    }
                });
            }

            const code = this.editor ? this.editor.getValue() : '';
            const language = this.getLanguageFromExtension(this.currentFile || 'script.js');

            // Check for RAG toggle
            const ragToggle = document.getElementById('rag-toggle');
            const useRag = ragToggle ? ragToggle.checked : false;

            const context = {
                current_code: code.substring(0, 1000),
                current_language: language,
                current_file: this.currentFile || 'Untitled',
                use_rag: useRag
            };

            // Detect Git requests or simple questions
            const isGitRequest = /git|push|pull|branch|commit|checkout|sync|status/i.test(message);
            const isSimpleQuestion = /how|what|why|who|explain|tell me/i.test(message) && message.length < 50;

            if (isGitRequest || isSimpleQuestion) {
                const response = await fetch('http://127.0.0.1:5000/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message,
                        history,
                        context,
                        use_rag: useRag
                    })
                });

                const result = await response.json();
                this.hideAITypingIndicator();

                if (result.success) {
                    this.addAIMessage('ai', result.response);

                    // Parse Git Actions if present
                    const gitActionMatch = result.response.match(/<git_action>([\s\S]*?)<\/git_action>/);
                    if (gitActionMatch) {
                        try {
                            const actionData = JSON.parse(gitActionMatch[1]);
                            // Reuse robust git logic (handles tokens, etc.)
                            this.handleGitAction(actionData.action, actionData.data);
                        } catch (e) {
                            console.error('Failed to parse git action:', e);
                        }
                    }
                } else {
                    this.addAIMessage('ai', `**Chat Error:** ${result.error}`);
                }
                return;
            }

            // Otherwise use the Multi-File Edit/Agent Workflow for complex code tasks
            this.hideAITypingIndicator();
            await this.handleMultiFileEdit(message);
            return;



        } catch (error) {
            this.hideAITypingIndicator();
            console.error('AI chat error:', error);
            this.addAIMessage('ai', `**Chat Error:** ${error.message}\n\nMake sure Flask backend is running: python app.py in backend directory`);
        }
    }

    async handleMultiFileEdit(task) {
        try {
            // Maxwell's Demon: Sorting the fast (in memory) from the slow (needs fetching)
            // 1. Collect potential context files
            let contextFiles = [];

            // Add current file
            if (this.currentFile && !this.currentFile.startsWith('Untitled')) {
                contextFiles.push(this.currentFile);
            }

            // Add open tabs
            for (const [filePath, tab] of this.tabs.entries()) {
                if (filePath && !filePath.startsWith('Untitled')) {
                    contextFiles.push(filePath);
                }
            }

            // Add visible files in root (simple heuristic for now, RAG will handle deep search)
            if (this.fileTree && this.fileTree.children) {
                this.fileTree.children.forEach(child => {
                    if (child.type === 'file' && !contextFiles.includes(child.path)) {
                        contextFiles.push(child.path);
                    }
                });
            }

            contextFiles = [...new Set(contextFiles)]; // Dedupe


            console.log('Sending multi-file edit request:', { task, files: contextFiles });

            this.addAIMessage('ai', `🧠 **Generating Plan & Edits...**\n\n*Analyzing ${contextFiles.length} files with RAG & Tree-sitter...*`);
            this.showAITypingIndicator();

            const result = await window.electronAPI.multiFileEdit(task, contextFiles);

            this.hideAITypingIndicator();

            if (result.success) {
                // Show Plan
                if (result.plan) {
                    this.addAIMessage('ai', `**Plan:**\n${result.plan}\n\nReviewing changes...`);
                }

                // Cursor-like Review: Open tabs and set them to Diff Mode
                if (result.edits && result.edits.length > 0) {
                    this.addAIMessage('ai', `I've proposed changes for ${result.edits.length} files. Opening them for review...`);

                    for (const edit of result.edits) {
                        // Normalize path immediately to match tab system
                        const filePath = this.normalizePath(edit.file_path);
                        const fileName = this.getFileNameFromPath(filePath);

                        // 1. Ensure Original Content exists
                        if (!edit.is_new && !edit.original_content) {
                            try {
                                const fs = require('fs');
                                if (fs.existsSync(edit.file_path)) {
                                    edit.original_content = fs.readFileSync(edit.file_path, 'utf-8');
                                }
                            } catch (e) {
                                console.error('Error reading original content:', e);
                                edit.original_content = ''; // Fallback
                            }
                        }

                        // 1.5 Apply Hunks Client-Side (if applicable)
                        if (edit.hunks && edit.hunks.length > 0 && edit.original_content) {
                            console.log(`Applying ${edit.hunks.length} hunks to ${fileName}`);
                            let content = edit.original_content;
                            let success = true;

                            // Sort hunks? Usually they come in order. simple sequential replace should work 
                            // if search blocks are unique enough.
                            for (const hunk of edit.hunks) {
                                // Normalize newlines in search block to match file system content if needed?
                                // Or just rely on string replacement
                                if (content.includes(hunk.search)) {
                                    content = content.replace(hunk.search, hunk.replace);
                                } else {
                                    console.warn(`Hunk failed: Could not find search block in ${fileName}`, hunk.search);
                                    success = false;
                                    // TODO: Visual warning?
                                }
                            }

                            edit.new_content = content;
                            edit.patchSuccess = success;
                        } else if (!edit.new_content && !edit.is_new) {
                            // If no new_content and no hunks, maybe no changes?
                            edit.new_content = edit.original_content;
                        }

                        // 2. Pre-configure Tab State for Diff View
                        // We set this BEFORE openFile so that when openFile calls setActiveTab,
                        // it renders the Diff View directly, avoiding double-rendering/flashing.

                        if (!this.tabs.has(filePath)) {
                            this.createTab(filePath, fileName);
                        }

                        const tabInfo = this.tabs.get(filePath);
                        if (tabInfo) {
                            tabInfo.isDiff = true;
                            tabInfo.editData = edit;
                            tabInfo.element.classList.add('review-mode');
                        }

                        // 3. Open the file (updates content and activates tab)
                        const contentToLoad = edit.is_new ? '' : (edit.original_content || '');
                        await this.openFile(filePath, contentToLoad, fileName);

                        // Note: openFile -> setActiveTab will see isDiff=true and render the Diff Editor.
                    }

                    // activate the first one again to ensure focus on the list
                    if (result.edits.length > 0) {
                        this.setActiveTab(result.edits[0].file_path);
                    }

                } else {
                    this.addAIMessage('ai', `Analyzed context but found no code changes needed.`);
                }
            } else {
                this.addAIMessage('ai', `**Error executing workflow:** ${result.error}`);
            }

        } catch (error) {
            this.hideAITypingIndicator();
            console.error('Workflow error:', error);
            this.addAIMessage('ai', `**Workflow Error:** ${error.message}`);
        }
    }

    showEditPreview(edits) {
        this.pendingEdits = edits;

        // Remove existing review panel if any
        const existingPanel = document.querySelector('.review-panel');
        if (existingPanel) existingPanel.remove();

        const messages = document.getElementById('ai-chat-messages');

        const panel = document.createElement('div');
        panel.className = 'review-panel';

        let fileListHtml = '';
        edits.forEach((edit, index) => {
            const fileName = this.getFileNameFromPath(edit.file_path);
            const status = edit.is_new ? 'New' : 'Modified';
            const statusClass = edit.is_new ? 'status-new' : 'status-modified';

            fileListHtml += `
                <div class="review-file-item" onclick="codeEditor.showDiff(${index})">
                    <i class="fas ${this.getFileIcon(fileName)}"></i>
                    <span style="flex:1">${fileName}</span>
                    <span class="file-status ${statusClass}">${status}</span>
                </div>
            `;
        });

        panel.innerHTML = `
            <div class="review-header">
                <div class="review-title">
                    <i class="fas fa-edit"></i> Review Changes (${edits.length})
                </div>
                <div class="review-actions">
                    <button class="btn-review btn-reject" onclick="codeEditor.rejectEdits()">
                        <i class="fas fa-times"></i> Reject
                    </button>
                    <button class="btn-review btn-accept" onclick="codeEditor.acceptEdits()">
                        <i class="fas fa-check"></i> Accept
                    </button>
                </div>
            </div>
            <div class="review-files">
                ${fileListHtml}
            </div>
        `;

        messages.appendChild(panel);
        messages.scrollTop = messages.scrollHeight;
    }

    async showDiff(editIndex) {
        const edit = this.pendingEdits[editIndex];
        if (!edit) return;

        // 1. Get Original Content
        let originalContent = '';
        if (!edit.is_new) {
            try {
                // Try reading from file system
                const result = await window.electronAPI.readFileContent(edit.file_path);
                originalContent = result;
            } catch (e) {
                console.warn('Could not read original file for diff:', e);
            }
        }

        const modifiedContent = edit.new_content;
        const language = this.getLanguageFromExtension(this.getFileNameFromPath(edit.file_path));

        // 2. Setup Diff Editor UI
        let diffContainer = document.getElementById('diff-editor-container');
        if (!diffContainer) {
            diffContainer = document.createElement('div');
            diffContainer.id = 'diff-editor-container';
            diffContainer.className = 'diff-editor-container';
            diffContainer.innerHTML = `
                <div class="diff-editor-header">
                    <span style="font-weight:600; margin-right:10px;">Diff: ${this.getFileNameFromPath(edit.file_path)}</span>
                    <span class="diff-editor-close" onclick="codeEditor.closeDiffView()">
                        <i class="fas fa-times"></i>
                    </span>
                </div>
                <div id="diff-monaco" class="diff-editor-content"></div>
            `;
            document.body.appendChild(diffContainer);
        }

        diffContainer.style.display = 'flex';

        // 3. Initialize Monaco Diff Editor
        if (this.diffEditor) {
            this.diffEditor.dispose();
        }

        require(['vs/editor/editor.main'], () => {
            const originalModel = monaco.editor.createModel(originalContent, language);
            const modifiedModel = monaco.editor.createModel(modifiedContent, language);

            this.diffEditor = monaco.editor.createDiffEditor(document.getElementById('diff-monaco'), {
                theme: 'vs-dark',
                readOnly: true,
                originalEditable: false,
                automaticLayout: true
            });

            this.diffEditor.setModel({
                original: originalModel,
                modified: modifiedModel
            });
        });
    }

    closeDiffView() {
        const diffContainer = document.getElementById('diff-editor-container');
        if (diffContainer) {
            diffContainer.style.display = 'none';
        }
        if (this.diffEditor) {
            this.diffEditor.dispose();
            this.diffEditor = null;
        }
    }

    async acceptEdits() {
        if (!this.pendingEdits || this.pendingEdits.length === 0) return;

        // Remove review panel
        const panel = document.querySelector('.review-panel');
        if (panel) panel.remove();

        this.closeDiffView();

        await this.applyAllEdits(this.pendingEdits);
        this.pendingEdits = [];
    }

    async applyAllEdits(edits) {
        if (!edits || edits.length === 0) return;

        // Show progress?
        this.showNotification(`Applying ${edits.length} edits...`, 'info');

        for (const edit of edits) {
            await this.applySingleEdit(edit);
        }

        this.showNotification('All edits applied successfully', 'success');
    }

    rejectEdits() {
        const panel = document.querySelector('.review-panel');
        if (panel) panel.remove();
        this.closeDiffView();
        this.pendingEdits = [];
        this.addAIMessage('ai', '❌ Changes rejected.');
    }

    async generateDocumentation(type) {
        const container = document.getElementById('docs-container');
        if (!container) return;

        if (type === 'file' && !this.currentFile) {
            this.showNotification('No file opened to document', 'warning');
            return;
        }

        container.innerHTML = `<div class="loading-state" style="padding: 20px; text-align: center;">
            <i class="fas fa-spinner fa-spin" style="font-size: 24px; color: var(--accent-blue);"></i>
            <p style="margin-top: 10px;">Generating ${type} documentation...</p>
        </div>`;

        try {
            // Gather File Tree Context manually if RAG fails
            let fileTreeStr = "";
            try {
                // Quick hack: if fileTree exists, walk it
                // Better: ask backend for tree or rely on what we have
                if (this.fileTree) {
                    // Simplistic tree dump
                    fileTreeStr = JSON.stringify(Object.keys(this.fileTree));
                }
            } catch (e) {
                console.warn("Could not serialize file tree");
            }

            const contextMessage = type === 'project' ?
                `Generate comprehensive documentation for the entire project. Here is a list of files visible in the editor:\n${fileTreeStr}\n\nPlease infer the project structure and purpose based on these filenames (and RAG context if available).` :
                `Generate detailed documentation for the current file: ${this.currentFile}\n\nCode:\n${this.editor.getValue()}`;

            const response = await fetch('http://127.0.0.1:5000/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: contextMessage,
                    history: [],
                    use_rag: type === 'project',
                    context: {
                        file_list: fileTreeStr // Explicitly pass file list
                    }
                })
            });

            const result = await response.json();
            if (result.success) {
                // Format the AI response for the docs container
                container.innerHTML = this.formatAIContent(result.response);
                this.showNotification('Documentation generated successfully', 'success');
            } else {
                container.innerHTML = `<div class="error-state" style="color: var(--accent-red); padding: 20px;">
                    <i class="fas fa-exclamation-triangle"></i> Failed: ${result.error}
                </div>`;
            }
        } catch (error) {
            console.error('Doc generation error:', error);
            container.innerHTML = `<div class="error-state" style="color: var(--accent-red); padding: 20px;">
                Error: ${error.message}
            </div>`;
        }
    }

    // Marketplace Logic
    async searchExtensions(query) {
        const list = document.getElementById('marketplace-list');
        if (!list) return;

        list.innerHTML = `<div style="text-align:center; padding: 20px;"><i class="fas fa-spinner fa-spin"></i> Searching Open VSX...</div>`;

        try {
            const apiUrl = query
                ? `https://open-vsx.org/api/-/search?query=${encodeURIComponent(query)}&size=20`
                : `https://open-vsx.org/api/-/search?query=theme&size=20&sortBy=relevance`; // Default query

            const response = await fetch(apiUrl);
            const data = await response.json();

            const extensions = data.extensions || [];

            if (extensions.length === 0) {
                list.innerHTML = `<div class="empty-state">No extensions found for "${query}"</div>`;
                return;
            }

            list.innerHTML = extensions.map(ext => {
                const icon = ext.iconUrl || 'https://raw.githubusercontent.com/microsoft/vscode-icons/master/icons/dark/plugin.svg'; // Fallback
                return `
                <div class="marketplace-item glass hover-lift" style="margin-bottom: 12px; padding: 12px; border-radius: 12px; display: flex; gap: 12px; align-items: start; cursor: pointer; border: 1px solid rgba(255,255,255,0.05);">
                    <img src="${icon}" style="width: 42px; height: 42px; border-radius: 8px; object-fit: cover; background: #2d2d2d;" onerror="this.src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg'">
                    <div style="flex: 1; min-width: 0;">
                        <div style="font-weight: 600; font-size: 13px; color: white; margin-bottom: 2px;">${ext.displayName || ext.name}</div>
                        <div style="font-size: 11px; opacity: 0.6; color: #ccc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 4px;">${ext.description}</div>
                        <div style="display: flex; gap: 8px; font-size: 10px; opacity: 0.5;">
                            <span><i class="fas fa-user"></i> ${ext.namespace}</span>
                            <span><i class="fas fa-download"></i> ${ext.downloadCount || 0}</span>
                        </div>
                    </div>
                    <button class="btn-primary" style="padding: 4px 10px; font-size: 11px; height: fit-content;">Install</button>
                </div>
                `;
            }).join('');

        } catch (error) {
            console.error('Marketplace error:', error);
            list.innerHTML = `<div class="error-state" style="padding: 20px;">Failed to load extensions: ${error.message}</div>`;
        }
    }

    toggleLivePreview() {
        const panel = document.getElementById('live-preview-panel');
        const btn = document.getElementById('live-preview-btn');
        if (!panel) return;

        if (panel.style.display === 'none') {
            panel.style.display = 'flex';
            btn.classList.add('active');
            btn.style.color = 'var(--accent-blue)';
            this.updateLivePreview();

            // Auto-update on change
            this.editor.onDidChangeModelContent(() => {
                if (panel.style.display !== 'none') {
                    this.updateLivePreview();
                }
            });
        } else {
            panel.style.display = 'none';
            btn.classList.remove('active');
            btn.style.color = '';
        }
    }

    updateLivePreview() {
        const iframe = document.getElementById('preview-iframe');
        if (!iframe) return;

        const content = this.editor.getValue();
        const language = monaco.editor.getModel(this.editor.getModel().uri).getLanguageId();

        if (['html', 'htm'].includes(language)) {
            iframe.srcdoc = content;
        } else if (['javascript', 'typescript'].includes(language)) {
            // Basic JS preview (wrap in script tags)
            iframe.srcdoc = `<html><body><script>${content}</script></body></html>`;
        } else if (['css'].includes(language)) {
            iframe.srcdoc = `<html><head><style>${content}</style></head><body><h1>CSS Preview</h1><p>The styles are applied to this preview context.</p></body></html>`;
        } else {
            iframe.srcdoc = `<html><body style="font-family: sans-serif; padding: 20px; color: #333;">
                <h3>Preview not available for ${language}</h3>
                <p>Open an HTML/JS/CSS file to see a live preview.</p>
            </body></html>`;
        }
    }

    async indexCodebase() {
        if (!this.workspacePath) {
            this.showNotification('Please open a folder first', 'warning');
            const ragToggle = document.getElementById('rag-toggle');
            if (ragToggle) ragToggle.checked = false;
            return;
        }

        // Visual feedback on the toggle label or notification
        this.showNotification('Indexing codebase...', 'info');
        const ragToggle = document.getElementById('rag-toggle');
        const label = document.querySelector('label[for="rag-toggle"]');
        const originalText = label ? label.textContent : 'Search Codebase';

        if (label) label.textContent = 'Indexing...';

        try {
            const result = await window.electronAPI.indexCodebase(this.workspacePath);

            if (result.success) {
                this.showNotification(`Indexed ${result.files} files`, 'success');
            } else {
                this.showNotification(`Indexing failed: ${result.error}`, 'error');
                if (ragToggle) ragToggle.checked = false;
            }
        } catch (error) {
            this.showNotification(`Indexing error: ${error.message}`, 'error');
            if (ragToggle) ragToggle.checked = false;
        } finally {
            if (label) label.textContent = originalText;
        }
    }

    showAITypingIndicator() {
        const messages = document.getElementById('ai-chat-messages');
        if (!messages) return;

        const typingIndicator = document.createElement('div');
        typingIndicator.id = 'ai-typing-indicator';
        typingIndicator.className = 'message ai-message fade-in';
        typingIndicator.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-sender">AI Assistant</div>
                <div class="message-text typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        messages.appendChild(typingIndicator);
        messages.scrollTop = messages.scrollHeight;
    }

    hideAITypingIndicator() {
        const typingIndicator = document.getElementById('ai-typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    addAIMessage(sender, content) {
        const messages = document.getElementById('ai-chat-messages');
        if (!messages) {
            console.error('AI chat messages element not found');
            return;
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message fade-in hover-lift`;

        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas ${sender === 'ai' ? 'fa-robot' : 'fa-user'}"></i>
            </div>
            <div class="message-content">
                <div class="message-sender">${sender === 'ai' ? 'AI Assistant' : 'You'}</div>
                <div class="message-text">${this.formatAIContent(content)}</div>
            </div>
        `;

        messages.appendChild(messageDiv);
        messages.scrollTop = messages.scrollHeight;

        if (sender === 'user') {
            this.aiChatHistory.push({ role: 'user', content: content });
        } else if (sender === 'ai') {
            this.aiChatHistory.push({ role: 'assistant', content: content });
        }
    }

    addChatAction(label, callback) {
        const messages = document.getElementById('ai-chat-messages');
        if (!messages) return;

        const actionDiv = document.createElement('div');
        actionDiv.className = 'chat-action fade-in';
        actionDiv.style.textAlign = 'center';
        actionDiv.style.margin = '10px 0';
        actionDiv.style.display = 'flex';
        actionDiv.style.justifyContent = 'center';

        const button = document.createElement('button');
        button.className = 'btn-primary';
        button.innerHTML = `<i class="fas fa-magic"></i> ${label}`;
        button.style.fontSize = '12px';
        button.style.padding = '8px 20px';
        button.style.borderRadius = '20px';
        button.style.border = 'none';
        button.style.boxShadow = '0 4px 15px rgba(0, 255, 255, 0.2)';
        button.style.transition = 'all 0.3s ease';

        button.onclick = async () => {
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Applying...';
            await callback();
            button.innerHTML = '<i class="fas fa-check"></i> Done';
            button.style.backgroundColor = '#4caf50'; // Green
        };

        actionDiv.appendChild(button);
        messages.appendChild(actionDiv);
        messages.scrollTop = messages.scrollHeight;
    }

    formatAIContent(content) {
        let formatted = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>');

        const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
        formatted = formatted.replace(codeBlockRegex, (match, lang, code) => {
            const language = lang || 'javascript';
            return `<div class="code-block" data-language="${language}">${this.escapeHtml(code)}</div>`;
        });

        return formatted;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    clearAIChat() {
        const messages = document.getElementById('ai-chat-messages');
        if (messages) {
            messages.innerHTML = `
                <div class="message ai-message">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <div class="message-sender">AI Assistant</div>
                        <div class="message-text">
                            Hello! I'm your AI coding assistant powered by LangChain. I can help you generate, explain, debug, and optimize code. How can I assist you today?
                        </div>
                    </div>
                </div>
            `;
            this.aiChatHistory = [];
        }
    }

    attachFileToChat() {
        const input = document.createElement('input');
        input.type = 'file';
        input.multiple = false;

        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const content = e.target.result;
                    this.addAIMessage('user', `**Attached file:** ${file.name}\n\n\`\`\`\n${content.substring(0, 1000)}\n\`\`\``);
                };
                reader.readAsText(file);
            }
        };

        input.click();
    }

    showAISettings() {
        this.showNotification('AI settings feature coming soon!', 'info');
    }

    updateOutline() {
        const outlineContent = document.getElementById('outline-content');
        if (!outlineContent) return;

        if (!this.editor) return;

        const content = this.editor.getValue();
        outlineContent.innerHTML = '';

        const lines = content.split('\n');
        let hasContent = false;

        lines.forEach((line, index) => {
            const trimmed = line.trim();

            if (trimmed.startsWith('function ') || trimmed.startsWith('const ') || trimmed.startsWith('let ') || trimmed.startsWith('var ') || trimmed.startsWith('class ')) {
                hasContent = true;
                const item = document.createElement('div');
                const type = trimmed.startsWith('function') ? 'function' :
                    trimmed.startsWith('class') ? 'class' : 'variable';

                item.className = `outline-item ${type} hover-lift`;
                const name = trimmed.split('(')[0].split('=')[0].replace(/function|const|let|var|class/g, '').trim();
                item.textContent = name || trimmed.substring(0, 30);
                item.title = `Line ${index + 1}: ${trimmed.substring(0, 50)}${trimmed.length > 50 ? '...' : ''}`;

                item.addEventListener('click', () => {
                    if (this.editor) {
                        this.editor.setPosition({ lineNumber: index + 1, column: 1 });
                        this.editor.focus();
                        this.editor.revealLineInCenter(index + 1);
                    }
                });

                outlineContent.appendChild(item);
            }
        });

        if (!hasContent) {
            outlineContent.innerHTML = '<div class="empty-outline">No symbols found</div>';
        }
    }

    updateTabStatus() {
        if (this.activeTab && this.tabs.has(this.activeTab)) {
            const tab = this.tabs.get(this.activeTab);
        }
    }

    getLanguageFromExtension(filename) {
        if (!filename) return 'plaintext';

        const ext = filename.split('.').pop().toLowerCase();
        const languageMap = {
            'js': 'javascript',
            'jsx': 'javascript',
            'ts': 'typescript',
            'tsx': 'typescript',
            'html': 'html',
            'htm': 'html',
            'css': 'css',
            'scss': 'scss',
            'sass': 'sass',
            'less': 'less',
            'py': 'python',
            'json': 'json',
            'md': 'markdown',
            'java': 'java',
            'c': 'c',
            'cpp': 'cpp',
            'h': 'c',
            'hpp': 'cpp',
            'cs': 'csharp',
            'php': 'php',
            'rb': 'ruby',
            'go': 'go',
            'rs': 'rust',
            'sql': 'sql',
            'xml': 'xml',
            'yaml': 'yaml',
            'yml': 'yaml',
            'vue': 'vue',
            'svelte': 'html'
        };
        return languageMap[ext] || 'plaintext';
    }

    getFileNameFromPath(path) {
        if (!path) return 'Untitled';
        const parts = path.split(/[\\/]/);
        return parts[parts.length - 1] || 'Workspace';
    }

    showFileContextMenu(e) {
        const contextMenu = document.getElementById('file-context-menu');
        if (!contextMenu) return;

        contextMenu.style.display = 'block';
        contextMenu.style.left = `${e.pageX}px`;
        contextMenu.style.top = `${e.pageY}px`;
        contextMenu.classList.add('glass');

        contextMenu.querySelectorAll('.context-menu-item').forEach(item => {
            item.onclick = (e) => {
                const action = e.currentTarget.dataset.action;
                this.handleContextMenuAction(action);
                contextMenu.style.display = 'none';
            };
        });
    }

    // showEditorContextMenu(e) {
    // }

    hideContextMenus() {
        const contextMenu = document.getElementById('file-context-menu');
        if (contextMenu) {
            contextMenu.style.display = 'none';
        }
    }

    handleContextMenuAction(action) {
        const targetPath = this.getContextMenuTarget();
        if (!targetPath) return;

        switch (action) {
            case 'open':
                break;
            case 'rename':
                if (targetPath.startsWith('/')) {
                    this.showNotification('Cannot rename default files', 'warning');
                    return;
                }
                const currentName = targetPath.split(/[\\/]/).pop();
                this.showRenameDialog(targetPath, currentName);
                break;
            case 'delete':
                this.deleteFile(targetPath);
                break;
            case 'copy-path':
                navigator.clipboard.writeText(targetPath);
                this.showNotification('Path copied to clipboard', 'success');
                break;
            case 'reveal-in-explorer':
                this.showNotification('Reveal in explorer feature coming soon!', 'info');
                break;
        }
    }

    getContextMenuTarget() {
        const activeItem = document.querySelector('.file-item.active, .folder-item.active');
        return activeItem ? activeItem.dataset.path : null;
    }

    handleMenuAction(action) {
        switch (action) {
            case 'new-file':
                this.createNewFile();
                break;
            case 'new-folder':
                this.createNewFolder();
                break;
            case 'save-file':
                this.saveCurrentFile();
                break;
            case 'save-file-as':
                this.saveFileAs();
                break;
            case 'run-code':
                this.runCurrentFile();
                break;
            case 'find':
                if (this.editor) {
                    this.editor.getAction('actions.find').run();
                }
                break;
            case 'replace':
                if (this.editor) {
                    this.editor.getAction('editor.action.startFindReplaceAction').run();
                }
                break;
            case 'toggle-comment':
                if (this.editor) {
                    this.editor.getAction('editor.action.commentLine').run();
                }
                break;
            case 'format-document':
                if (this.editor) {
                    this.editor.getAction('editor.action.formatDocument').run();
                }
                break;
            case 'shortcuts':
                this.showKeyboardShortcuts();
                break;
            case 'about':
                this.showAboutDialog();
                break;
        }
    }

    handleWindowAction(action) {
        switch (action) {
            case 'toggle-sidebar':
                const sidebar = document.getElementById('sidebar');
                if (sidebar) {
                    sidebar.classList.toggle('collapsed');
                }
                break;
            case 'toggle-ai-panel':
                this.toggleAIPanel();
                break;
        }
    }

    handleTerminalAction(action) {
        switch (action) {
            case 'new-terminal':
                this.showTerminal();
                break;
            case 'run-current':
                this.runCurrentFile();
                break;
        }
    }

    handleKeyboardShortcuts(e) {
        if (!document.activeElement.closest('.monaco-editor')) {
            return;
        }

        if (e.ctrlKey || e.metaKey) {
            switch (e.key.toLowerCase()) {
                case 's':
                    e.preventDefault();
                    this.saveCurrentFile();
                    break;
                case 'n':
                    e.preventDefault();
                    this.createNewFile();
                    break;
                case 'o':
                    e.preventDefault();
                    this.openFileDialog();
                    break;
                case 'f':
                    e.preventDefault();
                    if (this.editor) {
                        this.editor.getAction('actions.find').run();
                    }
                    break;
                case 'b':
                    e.preventDefault();
                    const sidebar = document.getElementById('sidebar');
                    if (sidebar) {
                        sidebar.classList.toggle('collapsed');
                    }
                    break;
                case 'i':
                    e.preventDefault();
                    this.toggleAIPanel();
                    break;
                case '/':
                    e.preventDefault();
                    if (this.editor) {
                        this.editor.getAction('editor.action.commentLine').run();
                    }
                    break;
                case 'shift':
                    if (e.key === 'o' && e.ctrlKey) {
                        e.preventDefault();
                        this.openFolderDialog();
                        break;
                    }
            }
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type} fade-in glass`;
        notification.textContent = message;

        notification.style.cssText = `
            position: fixed;
            top: 60px;
            right: 20px;
            padding: 14px 20px;
            background: ${type === 'success' ? '#27ca3f' :
                type === 'error' ? '#ff5f56' :
                    type === 'warning' ? '#ffbd2e' : '#007acc'};
            color: white;
            border-radius: 12px;
            z-index: 10000;
            box-shadow: var(--shadow-lg);
            font-size: 13px;
            max-width: 350px;
            word-wrap: break-word;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        `;

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(20px)';
            notification.style.transition = 'all 0.3s ease';

            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    showKeyboardShortcuts() {
        const shortcuts = [
            { key: 'Ctrl+S', action: 'Save File' },
            { key: 'Ctrl+N', action: 'New File' },
            { key: 'Ctrl+O', action: 'Open File' },
            { key: 'Ctrl+Shift+O', action: 'Open Folder' },
            { key: 'Ctrl+B', action: 'Toggle Sidebar' },
            { key: 'Ctrl+Shift+I', action: 'Toggle AI Panel' },
            { key: 'F5', action: 'Run Code' },
            { key: 'Ctrl+F', action: 'Find' },
            { key: 'Ctrl+H', action: 'Replace' },
            { key: 'Ctrl+/', action: 'Toggle Comment' },
            { key: 'Shift+Alt+F', action: 'Format Document' }
        ];

        const message = shortcuts.map(s => `${s.key.padEnd(20)} ${s.action}`).join('\n');
        alert('Keyboard Shortcuts:\n\n' + message);
    }

    showAboutDialog() {
        const dialog = document.createElement('div');
        dialog.className = 'dialog glass';
        dialog.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        `;

        dialog.innerHTML = `
            <div class="dialog-content" style="
                background: var(--glass-bg);
                backdrop-filter: blur(20px);
                border: 1px solid var(--glass-border);
                border-radius: 16px;
                padding: 30px;
                min-width: 400px;
                max-width: 500px;
                box-shadow: var(--shadow-lg);
            ">
                <h3 style="
                    margin: 0 0 20px 0;
                    color: var(--text-primary);
                    font-size: 24px;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                ">
                    <i class="fas fa-code" style="color: var(--accent-blue);"></i>
                    AI Code Editor
                </h3>
                <div style="color: var(--text-secondary); margin-bottom: 20px; line-height: 1.6;">
                    <p><strong>Version:</strong> 1.0.0</p>
                    <p>A VS Code-like editor with AI assistance</p>
                    <p>Built with Electron, Monaco Editor, and Flask</p>
                </div>
                <div style="margin-bottom: 20px;">
                    <h4 style="color: var(--text-primary); margin-bottom: 10px;">Features:</h4>
                    <ul style="color: var(--text-secondary); padding-left: 20px; line-height: 1.6;">
                        <li>Code editing with syntax highlighting</li>
                        <li>File management with real-time watching</li>
                        <li>AI-powered code assistance</li>
                        <li>Terminal emulator</li>
                        <li>VS Code-like interface</li>
                        <li>Auto-save functionality</li>
                        <li>Modern glassmorphism design</li>
                    </ul>
                </div>
                <div style="text-align: center;">
                    <button class="btn-modern" onclick="this.closest('.dialog').remove()" style="padding: 10px 30px;">
                        Close
                    </button>
                </div>
            </div>
        `;

        dialog.addEventListener('click', (e) => {
            if (e.target === dialog) {
                dialog.remove();
            }
        });

        document.body.appendChild(dialog);
    }

    splitEditor() {
        this.showNotification('Split editor feature coming soon!', 'info');
    }

    showMenu(menuName, target) {
        switch (menuName) {
            case 'file':
                this.showNotification('File menu - Use submenu items', 'info');
                break;
            case 'edit':
                this.showNotification('Edit menu - Use submenu items', 'info');
                break;
            case 'view':
                this.showNotification('View menu - Use submenu items', 'info');
                break;
            case 'ai':
                this.toggleAIPanel();
                break;
            case 'terminal':
                this.showTerminal();
                break;
            case 'help':
                this.showAboutDialog();
                break;
        }
    }

    openFileDialog() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.js,.ts,.jsx,.tsx,.html,.css,.py,.json,.md,.txt,.xml,.yaml,.yml,.java,.c,.cpp,.h,.cs,.php,.rb,.go,.rs,.sql,.vue,.svelte';

        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const content = e.target.result;
                    this.openFile(file.path, content, file.name);
                    this.showNotification(`Opened file: ${file.name}`, 'success');
                };
                reader.readAsText(file);
            }
        };

        input.click();
    }

    async openFolderDialog() {
        try {
            const result = await window.electronAPI.showOpenFolderDialog();
            if (!result.canceled && result.filePaths && result.filePaths.length > 0) {
                const folderPath = result.filePaths[0];
                await this.openFolderFromPath(folderPath);
            }
        } catch (error) {
            console.error('Error opening folder:', error);
            this.showNotification(`Error opening folder: ${error.message}`, 'error');
        }
    }

    updateWorkspaceInfo(folderPath) {
        let workspaceInfo = document.getElementById('workspace-info');
        if (!workspaceInfo) {
            workspaceInfo = document.createElement('div');
            workspaceInfo.id = 'workspace-info';
            workspaceInfo.className = 'workspace-indicator glass';

            const workspaceElement = document.getElementById('workspace');
            if (workspaceElement) {
                workspaceElement.insertBefore(workspaceInfo, workspaceElement.firstChild);
            }
        }

        const folderName = this.getFileNameFromPath(folderPath);
        workspaceInfo.innerHTML = `
            <i class="fas fa-folder-open" style="color: var(--accent-blue);"></i>
            <span style="margin-left: 8px; font-weight: 500;">${folderName}</span>
            <span style="margin-left: 8px; font-size: 10px; opacity: 0.7;">(${folderPath})</span>
            <div style="margin-left: auto; display: flex; gap: 8px;">
                <i class="fas fa-sync-alt" style="cursor: pointer; font-size: 11px;" 
                   onclick="codeEditor.refreshFileTree()" title="Refresh"></i>
                <i class="fas fa-times" style="cursor: pointer; font-size: 11px;" 
                   onclick="codeEditor.closeWorkspace()" title="Close Workspace"></i>
            </div>
        `;
    }

    updateFileInfo(filePath) {
        if (filePath && !filePath.startsWith('/')) {
            try {
                const fs = require('fs');
                const stats = fs.statSync(filePath);
                const size = this.formatFileSize(stats.size);
                const modified = new Date(stats.mtime).toLocaleString();

                const fileInfoElement = document.createElement('div');
                fileInfoElement.className = 'status-item';
                fileInfoElement.innerHTML = `<i class="fas fa-info-circle"></i><span>${size} • ${modified}</span>`;

                const statusRight = document.querySelector('.status-right');
                if (statusRight) {
                    const existingInfo = statusRight.querySelector('.file-info-status');
                    if (existingInfo) {
                        existingInfo.replaceWith(fileInfoElement);
                    } else {
                        fileInfoElement.classList.add('file-info-status');
                        statusRight.prepend(fileInfoElement);
                    }
                }
            } catch (error) {
                console.error('Error getting file info:', error);
            }
        }
    }

    closeWorkspace() {
        this.workspacePath = null;
        this.fileTree = { children: [] };

        this.fileWatchers.forEach(watcher => watcher.close());
        this.fileWatchers.clear();

        this.renderFileTree();

        const workspaceInfo = document.getElementById('workspace-info');
        if (workspaceInfo) {
            workspaceInfo.remove();
        }

        const fileInfoStatus = document.querySelector('.file-info-status');
        if (fileInfoStatus) {
            fileInfoStatus.remove();
        }

        this.showNotification('Workspace closed', 'info');
    }

    // Cleanup method
    cleanup() {
        this.fileSystemWatchers.forEach(watcher => watcher.close());
        this.fileSystemWatchers.clear();

        this.fileWatchers.forEach(watcher => watcher.close());
        this.fileWatchers.clear();

        if (this.editor) {
            this.editor.dispose();
        }
    }
    addToExplorer(filePath, fileName) {
        console.log(`Adding to explorer: ${fileName} (${filePath})`);
        if (!this.fileTree) {
            this.fileTree = {
                name: 'Files',
                path: '/',
                type: 'folder',
                open: true,
                children: []
            };
        }

        if (this.isFileInTree(this.fileTree, filePath)) {
            console.log(`File already in tree: ${filePath}`);
            return;
        }

        console.log(`Adding new file to tree root: ${filePath}`);
        const newFile = {
            name: fileName,
            path: filePath,
            type: 'file'
        };

        if (!this.fileTree.children) {
            this.fileTree.children = [];
        }

        this.fileTree.children.push(newFile);
        this.renderFileTree();
    }

    isFileInTree(node, filePath) {
        // Normalize paths for comparison (handle Windows/Unix separators)
        const normalize = (p) => p ? p.replace(/\\/g, '/').toLowerCase() : '';
        const nodePath = normalize(node.path);
        const targetPath = normalize(filePath);

        if (nodePath === targetPath) return true;

        if (node.children) {
            return node.children.some(child => this.isFileInTree(child, filePath));
        }
        return false;
    }
}

window.addEventListener('beforeunload', () => {
    if (window.codeEditor) {
        window.codeEditor.cleanup();
    }

});