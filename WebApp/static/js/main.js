// Connexion au namespace 'reseausimple'
var socket = io.connect('http://' + document.domain + ':' + location.port + '/reseausimple');

function actualiserImage(imagePath) {
    var imageDisplay = document.getElementById('image-display');
    var nouvelleImage = document.createElement('img');
    nouvelleImage.src = imagePath;
    nouvelleImage.style.position = 'absolute';
    nouvelleImage.style.top = 0;
    nouvelleImage.style.left = 0;
    nouvelleImage.style.zIndex = imageDisplay.childNodes.length+1;
    nouvelleImage.style.width = '100%';
    if (imageDisplay.firstChild === null){
					imageDisplay.appendChild(nouvelleImage);
				}
				else{
					imageDisplay.insertBefore(nouvelleImage,imageDisplay.firstChild);
                }
    while (imageDisplay.childNodes.length > 2) {
        imageDisplay.removeChild(imageDisplay.lastChild);
    }
}

function actualiserBarre(avancement) {
    var barre = document.getElementById('loading-bar');
    var pourcentage = avancement*100
    barre.style.width = pourcentage+'%';
}

document.addEventListener('DOMContentLoaded', function() {
    socket.on('nouvelle_image', function(data) {
        actualiserImage(data.image_path);
    });
    socket.on('avancement', function(data) {
        actualiserBarre(data);
    });
});

function startTraining() {
    socket.emit('update_image');
    return false;  // Empêche le formulaire de recharger la page
}
