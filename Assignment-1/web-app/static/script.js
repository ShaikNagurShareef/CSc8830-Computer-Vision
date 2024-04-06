function processImage() {
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');
    const canvas = document.getElementById('imageCanvas');
    const ctx = canvas.getContext('2d');

    const file = fileInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(event) {
        const img = new Image();
        img.onload = function() {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            canvas.style.display = 'block';
            const imageData = canvas.toDataURL('image/png');
            fetch('/process_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            })
            .then(response => response.json())
            .then(data => {
                resultDiv.innerText = `Diameter of circular object is: ${data.distance} mm`;
            })
            .catch(error => console.error('Error:', error));
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}
