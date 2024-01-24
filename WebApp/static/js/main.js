// Connexion au namespace 'reseausimple'
var socket = io.connect('http://' + document.domain + ':' + location.port + '/reseausimple');

function actualiserImage(imagePath) {
    var imageDisplay = document.getElementById('image-display');
    var nouvelleImage = document.createElement('img');
    nouvelleImage.src = imagePath;
    if (imageDisplay.firstChild === null){
                    nouvelleImage.style.position = 'absolute';
					imageDisplay.appendChild(nouvelleImage);
				}
				else{
					imageDisplay.insertBefore(nouvelleImage,imageDisplay.firstChild)
                }
    while (imageDisplay.childNodes.length > 2) {
        imageDisplay.removeChild(imageDisplay.lastChild);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    socket.on('nouvelle_image', function(data) {
        actualiserImage(data.image_path);
    });
});

function startTraining() {
    socket.emit('update_image');
    return false;  // Empêche le formulaire de recharger la page
}
