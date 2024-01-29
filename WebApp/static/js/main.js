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
    var barre = document.getElementById('loading-bar2');
    barre.style.width = avancement;
    var etape = document.getElementById('affichage_etape');
    etape.textContent = avancement;
}

function actualiserReseau(weights) {
    Object.keys(weights).forEach(function(cle) {weights[cle] = Math.abs(weights[cle]);});
    var wmax = Math.max(...Object.values(weights));
    var wmin = Math.min(...Object.values(weights));
    var echelle = 30/(wmax - wmin)
    for (var w in weights) {
        document.getElementById(w).style.strokeWidth = parseInt(Math.abs((weights[w]-wmin)*echelle + 10));
    }
    console.log(document.getElementById('w020').style.strokeWidth)
}

document.addEventListener('DOMContentLoaded', function() {
    socket.on('nouvelle_image', function(data) {
        actualiserImage(data.image_path);
    });
    socket.on('avancement', function(data) {
        actualiserBarre(data);
    });
    socket.on('update_net', function(data) {
        actualiserReseau(data);
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

function showImageN(n) {
    document.getElementById('affichage_etape').textContent = n;
    socket.emit('get_image', {etape:n});
}

const fond = document.getElementById('loading-bar-container');
const barre = document.getElementById('loading-bar2');

let isDragging = false;
let nombrePointsSnap = 0;

fond.addEventListener('mousedown', (e) => {
    isDragging = true;
    fond.style.userSelect = 'none';
    fond.style.height = '10px';
});

document.addEventListener('mousemove', (e) => {
    if (isDragging) {
        const mouseX = e.clientX - fond.getBoundingClientRect().left;
        const pointSnap = Math.round((mouseX / fond.clientWidth) * nombrePointsSnap);
        console.log(pointSnap)
        let new_width = Math.max(0, Math.min(100, pointSnap * 100 / nombrePointsSnap))
        if (new_width !=== barre.offsetWidth) {
            barre.style.width = new_width + '%';
            showImageN(parseInt(barre.offsetWidth / (100 / nombrePointsSnap))
        }
    }
});

document.addEventListener('mouseup', () => {
    if (isDragging) {
        isDragging = false;
        fond.style.userSelect = '';
        fond.style.height = '5px';
    }
});

window.onbeforeunload = function() {
    socket.emit('closing_page');
};