// Connexion au namespace 'reseausimple'
var socket = io.connect('http://' + document.domain + ':' + location.port + '/reseausimple');

function actualiserImage(imagePath) {
    var imageDisplay = document.getElementById('image-display');
    var nouvelleImage = document.createElement('img');
    nouvelleImage.src = imagePath;
    nouvelleImage.style.position = 'absolute';
    nouvelleImage.style.top = 0;
    nouvelleImage.style.left = 0;
    nouvelleImage.style.width = '100%';
    nouvelleImage.style.zIndex = 1;
    if (imageDisplay.firstChild === null){
					imageDisplay.appendChild(nouvelleImage);
				}
				else{
					imageDisplay.insertBefore(nouvelleImage,imageDisplay.firstChild);
                }
    while (imageDisplay.childNodes.length > 2) {
        imageDisplay.removeChild(imageDisplay.lastChild);
    }
    imageDisplay.lastChild.style.zIndex = 0;
}

function actualiserBarre(avancement) {
    var barre = document.getElementById('loading-bar');
    barre.value = avancement;
    var etape = document.getElementById('affichage_etape');
    etape.textContent = avancement;
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
    var nombre_passages = parseInt(document.getElementById('nombre_passages').value, 10);
    var barre = document.getElementById('loading-bar');
    barre.max = parseInt(barre.max) + nombre_passages;
    socket.emit('start_training', {passages:nombre_passages});
    return false;  // Empêche le formulaire de recharger la page
}

function restartTraining() {
    var barre = document.getElementById('loading-bar');
    barre.max = 0;
    barre.value = 0;
    var etape = document.getElementById('affichage_etape');
    etape.textContent = 0;
    socket.emit('resume_training');
}

function showImageN() {
    var n = parseInt(document.getElementById('loading-bar').value, 10);
    var etape = document.getElementById('affichage_etape');
    etape.textContent = n;
    socket.emit('get_image', {etape:n});
}

window.onbeforeunload = function() {
    socket.emit('closing_page');
};