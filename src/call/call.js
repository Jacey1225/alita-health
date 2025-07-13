const toggleBtn = document.getElementById('toggleChat');
const closeBtn = document.getElementById('closeChat');
const hiddenBtn = document.getElementById('hidden-button')

toggleBtn.addEventListener('click', () => {
    document.body.classList.add('chat-active');
    toggleBtn.classList.add('hidden');
});

closeBtn.addEventListener('click', () => {
    document.body.classList.remove('chat-active');
    toggleBtn.classList.remove('hidden');
});