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

typeWriter("input", "type here...", 100, true, 0);
typeWriter("type-two", "CHAT WITH ALITA!", 100, false, 0);
