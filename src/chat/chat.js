// Typewriter animation function
function typeWriter(id, text, speed, isPlaceholder = false, delay = 0) {
  let i = 0;
  const target = document.getElementById(id);

  setTimeout(() => {
    function typing() {
      if (i < text.length) {
        if (isPlaceholder) {
          target.placeholder += text.charAt(i);
        } else {
          target.textContent += text.charAt(i);
        }
        i++;
        setTimeout(typing, speed);
      }
    }

    // Clear existing content
    if (isPlaceholder) {
      target.placeholder = '';
    } else {
      target.textContent = '';
    }

    typing();
  }, delay);
}

// Start typing animations on page load
window.onload = () => {
  typeWriter("input", "type here...", 100, true);
  typeWriter("type-two", "CHAT WITH ALITA!", 100, false);
};

// Main chat logic
document.getElementById('sendBtn').addEventListener('click', sendMessage);

async function sendMessage() {
  const inputField = document.getElementById('input');
  const chatDisplay = document.getElementById('chat-display');
  const message = inputField.value.trim();

  if (!message) return;

  // Display user's message
  const userDiv = document.createElement('div');
  userDiv.className = 'chat-message user-msg';
  userDiv.textContent = `You: ${message}`;
  chatDisplay.appendChild(userDiv);

  inputField.value = '';

  try {
    console.log('Sending request to:', 'http://localhost:5001/generate');
    console.log('Message:', message);

    const response = await fetch('http://localhost:5001/generate', {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ message })
    });

    console.log('Response status:', response.status);
    console.log('Response ok:', response.ok);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Response error:', errorText);
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    console.log('Response data:', data);

    if (data.error) {
      throw new Error(data.error);
    }

    const botDiv = document.createElement('div');
    botDiv.className = 'chat-message bot-msg';
    botDiv.textContent = `ALITA: ${data.response || 'No response generated'}`;
    chatDisplay.appendChild(botDiv);
    
  } catch (err) {
    console.error('Fetch error details:', err);
    const errDiv = document.createElement('div');
    errDiv.className = 'chat-message bot-msg';
    errDiv.textContent = `ALITA: Sorry, something went wrong. ${err.message}`;
    chatDisplay.appendChild(errDiv);
  }

  chatDisplay.scrollTop = chatDisplay.scrollHeight;
}
