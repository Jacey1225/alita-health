function setupImageSwitcher(elementId, initialSrc, alternateSrc) {
  const image = document.getElementById(elementId);
  let isOriginal = true;

  if (!image) return;

  image.src = initialSrc;
  image.alt = "Original Image";

  image.addEventListener('click', () => {
    image.src = isOriginal ? alternateSrc : initialSrc;
    image.alt = isOriginal ? "Alternate Image" : "Original Image";
    isOriginal = !isOriginal;
  });
}

window.onload = function () {
  setupImageSwitcher(
    "theme-toggle-1",
    "../../assets/light-mode.png",
    "../../assets/dark-mode.png"
  );

  setupImageSwitcher(
    "theme-toggle-2",
    "../../assets/light-mode.png",
    "../../assets/dark-mode.png"
  );
};
