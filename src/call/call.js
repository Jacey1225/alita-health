const toggleBtn = document.getElementById('toggleChat');
const closeBtn = document.getElementById('closeChat');
const hiddenBtn = document.getElementById('hidden-button')
const text = "TRANSCRIPTION";
const speed = 100; // milliseconds per character
let i = 0;

function typeWriter() {
  if (i < text.length) {
    document.getElementById("typewriter").textContent += text.charAt(i);
    i++;
    setTimeout(typeWriter, speed);
  }
}

toggleBtn.addEventListener('click', () => {
    document.body.classList.add('chat-active');
    toggleBtn.classList.remove('hidden');
    document.getElementById("typewriter").textContent = "";
    i = 0;
    typeWriter();
});

closeBtn.addEventListener('click', () => {
    document.body.classList.remove('chat-active');
    toggleBtn.classList.remove('hidden');
});