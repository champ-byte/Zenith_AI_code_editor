// --- 1. Mock Data (The File System) ---
const projectFiles = [
    "index.html",
    "style.css",
    "app.js",
    "components/navbar.js",
    "utils/helpers.js",
    "readme.md"
];

// --- 2. Setup Speech Recognition ---
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
recognition.lang = "en-US";
recognition.interimResults = false;

// --- 3. The "Normalizer" Helper ---
// Converts spoken words into code-friendly symbols
function normalizeText(text) {
    return text
        .toLowerCase()
        .replace(/ dot /g, ".")   // "index dot html" -> "index.html"
        .replace(/ slash /g, "/") // "components slash navbar" -> "components/navbar"
        .trim();
}

// --- 4. Handle Results ---
recognition.onresult = function(event) {
    // A. Get raw text
    const rawTranscript = event.results[0][0].transcript;
    
    // B. Normalize it (fix dots/slashes)
    const cleanTranscript = normalizeText(rawTranscript);
    
    // C. Update UI
    document.getElementById("output").innerText = `Raw: "${rawTranscript}" \nProcessed: "${cleanTranscript}"`;

    // D. Check for commands
    processCommand(cleanTranscript);
};

// --- 5. The Command Processor ---
function processCommand(command) {
    // Check if the user wants to "open" a file
    if (command.startsWith("open")) {
        
        // Remove the word "open" to get the filename (e.g., "open style.css" -> "style.css")
        const searchFile = command.replace("open", "").trim();
        
        // Search the file list
        const foundFile = projectFiles.find(file => file.includes(searchFile));

        if (foundFile) {
            alert(`✅ Opening file: ${foundFile}`);
            // In the real app, this is where you would trigger your tab opening logic
        } else {
            alert(`❌ Could not find a file matching: "${searchFile}"`);
        }
    }
}

// --- 6. Start listening ---
function startListening() {
    recognition.start();
    document.getElementById("output").innerText = "Listening...";
}