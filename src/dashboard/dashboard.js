const text = "WELCOME, USERNAME!";
const speed = 100; // milliseconds per character
let i = 0;

function typeWriter() {
  if (i < text.length) {
    document.getElementById("typewriter").textContent += text.charAt(i);
    i++;
    setTimeout(typeWriter, speed);
  }
}

typeWriter();

async function sendInput() {
  const input = document.getElementById('userInput').value;

  const response = await fetch('/get-response', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ message: input })
  });

  const data = await response.json();
  document.getElementById('responseText').textContent = data.reply;
}
