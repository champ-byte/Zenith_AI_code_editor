    // Speech Recognition setup
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;

    // Handle results
    recognition.onresult = function(event) {
      const transcript = event.results[0][0].transcript.toLowerCase();
      document.getElementById("output").innerText = "You said: " + transcript;
    console.log("Transcript: ", event.results);
    };

    // Start listening
    function startListening() {
      recognition.start();
    }