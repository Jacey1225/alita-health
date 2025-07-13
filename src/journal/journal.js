function typeWriter(id, text, speed, delay = 0) {
  let i = 0;
  const target = document.getElementById(id);
  target.textContent = ''; // Clear previous content

  setTimeout(() => {
    function typing() {
      if (i < text.length) {
        target.textContent += text.charAt(i);
        i++;
        setTimeout(typing, speed);
      }
    }
    typing();
  }, delay);
}

typeWriter("type-one", "How are you feeling today?", 100);
typeWriter("type-two", "Your quote of the day is:", 100, 3000);
typeWriter("type-three", "'You are enough'", 100, 6000);

const toggleBtn = document.getElementById('toggleEntry');
const closeBtn = document.getElementById('closeEntry');

toggleBtn.addEventListener('click', () => {
    document.body.classList.add('chat-active');
    toggleBtn.classList.add('hidden');
});

closeBtn.addEventListener('click', () => {
    document.body.classList.remove('chat-active');
    toggleBtn.classList.remove('hidden');
});

const emojis = document.querySelectorAll('.emojis');

emojis.forEach(emoji => {
    emoji.addEventListener('click', () => {
        // Remove selection from all
        emojis.forEach(e => e.classList.remove('emoji-selected'));

        // Add selection to clicked one
        emoji.classList.add('emoji-selected');
    });
});
