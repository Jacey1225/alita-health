// const text = "WELCOME, USERNAME!";
// const speed = 100; // milliseconds per character
// let i = 0;

// function typeWriter() {
//   if (i < text.length) {
//     document.getElementById("typewriter").textContent += text.charAt(i);
//     i++;
//     setTimeout(typeWriter, speed);
//   }
// }

// typeWriter();

// // dashboard.js
// window.onload = () => {
//   const username = localStorage.getItem('username');
//   const nameSpan = document.getElementById('username-span');
//   if (username && nameSpan) {
//     nameSpan.textContent = username.toUpperCase(); // Optional styling
//   }
// };


window.onload = () => {
  const username = localStorage.getItem('username') || 'Jacey';
  const typewriterElement = document.getElementById('typewriter');
  const fullText = `WELCOME, ${username.toUpperCase()}!`;
  let i = 0;

  function typeWriter() {
    if (i < fullText.length) {
      typewriterElement.textContent += fullText.charAt(i);
      i++;
      setTimeout(typeWriter, 100); // speed
    }
  }

  typeWriter();
};

