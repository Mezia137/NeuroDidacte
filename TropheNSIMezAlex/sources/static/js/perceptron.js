// Connexion au namespace 'reseausimple'
const socket = io.connect('http://' + document.domain + ':' + location.port + '/perceptron');
socket.emit('init')

const barContainer = document.getElementById('loading-bar-container');
const bar = document.getElementById('loading-bar');

const pp_button = document.getElementById("previouspage-link");
const np_button = document.getElementById("nextpage-link");


let isTraining = false;

let isDraggingBar = false;
let totalSteps = 0;
let step = 0;

function updateImage(imagePath) {
    var imageDisplay = document.getElementById('image-display');
    var nouvelleImage = document.createElement('img');
    nouvelleImage.src = imagePath;
    nouvelleImage.classList.add('image')
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

function updateBar(toStep) {
    if (totalSteps === 0) {
            bar.style.width = '0';
    } else {
            let new_width = toStep * 100 / totalSteps;
            bar.style.width = new_width + '%';
    }
    document.getElementById('affichage_etape').textContent = toStep;
}

function updateNet(weights) {
    Object.keys(weights).forEach(function(cle) {weights[cle] = Math.abs(weights[cle]);});
    var wmax = Math.max(...Object.values(weights));
    var wmin = Math.min(...Object.values(weights));
    var echelle = 30/(wmax - wmin)
    for (var w in weights) {
        document.getElementById(w).style.strokeWidth = parseInt(Math.abs((weights[w]-wmin)*echelle + 10));
    }
}

document.addEventListener('DOMContentLoaded', function() {
    socket.on('init_bar', function(data) {
        totalSteps = data['life'];
        updateBar(data['age']);
    });
    socket.on('nouvelle_image', function(data) {
        updateImage(data.image_path);
    });
    socket.on('avancement', function(data) {
        updateBar(data);
        if (data === totalSteps) {
            reableButtons();
        }
    });
    socket.on('update_net', function(data) {
        updateNet(data);
    });
});

function startTraining() {
    disableButtons()
    var nombre_passages = parseInt(document.getElementById('input-nombre_passages').value, 10);
    totalSteps += nombre_passages;
    socket.emit('training', {passages:nombre_passages});
    return false;
}

function restartTraining() {
    document.querySelectorAll('.ligne').forEach(function(element) {element.style.strokeWidth = '20px';});
    document.getElementById('image-display').innerHTML = '';
    socket.emit('resume_training');
}

function showImageN(n) {
    socket.emit('update', {age:n});
}

function disableButtons() {
    isTraining = true;
    document.querySelectorAll('button').forEach(function(bouton) {bouton.disabled = true;});
    document.querySelectorAll('.round-button').forEach(function(bouton) {bouton.classList.add('disabled_button')});
    document.getElementById("input-nombre_passages").disabled = true;

    document.getElementById("loading-bar-container").style.cursor = "not-allowed";

    pp_button.style.cursor = "not-allowed";
    pp_button.style.opacity = "0.2";
    pp_button.classList.add('disabled_button');
    
    np_button.style.cursor = "not-allowed";
    np_button.style.opacity = "0.2";
    np_button.classList.add('disabled_button');

    Array.from(document.links).forEach(link => {link.onclick = function(){return false;};});

    document.querySelector('header').style.cursor = 'not-allowed';
}

function reableButtons() {
    isTraining = false;
    document.querySelectorAll('button').forEach(function(bouton) {bouton.disabled = false;});
    document.querySelectorAll('.round-button').forEach(function(bouton) {bouton.classList.remove('disabled_button')});
    document.getElementById("input-nombre_passages").disabled = false;

    document.getElementById("loading-bar-container").style.cursor = "pointer";

    pp_button.style.cursor = "pointer";
    pp_button.style.opacity = "1";
    pp_button.classList.remove('disabled_button');

    np_button.style.cursor = "pointer";
    np_button.style.opacity = "1";
    np_button.classList.remove('disabled_button');

    Array.from(document.links).forEach(link => {link.onclick = null;});

    document.querySelector('header').style.cursor = 'auto';
}

barContainer.addEventListener('mousedown', (e) => {
    if (isTraining === false) {
        isDraggingBar = true;
        barContainer.style.userSelect = 'none';
        barContainer.style.height = '10px';
    }
});

document.addEventListener('mousemove', (e) => {
    if (isDraggingBar) {
        let mouseX = e.clientX - barContainer.getBoundingClientRect().left;
        oldstep = step;
        step = Math.round((Math.min(Math.max(mouseX, 0), window.innerWidth) / barContainer.clientWidth) * totalSteps);
        updateBar(step);
        if (oldstep != step) {
            showImageN(step)
            if (step === 0) {
                document.querySelectorAll('.ligne').forEach(function(element) {element.style.strokeWidth = '20px';});
            }
        }
    }
});

document.addEventListener('mouseup', () => {
    if (isDraggingBar) {
        isDraggingBar = false;
        barContainer.style.userSelect = '';
        barContainer.style.height = '5px';
    }
});


document.querySelectorAll('.info-icon').forEach(function(icon) {
    if (icon.dataset.infoboxId != "null") {
        icon.addEventListener('click', function(event) {
    const infoboxId = icon.dataset.infoboxId;
    const infobox = document.getElementById(infoboxId);

    cx = event.clientX;
    cy = event.clientY;

    if (cx > (window.innerWidth - 400)){
        infobox.style.left = cx-300 + 'px';
        infobox.style.top = cy+10 + 'px';
    } else {
        infobox.style.left = cx + 'px';
        infobox.style.top = cy+10 + 'px';
    }

    infobox.classList.add('show');
  });

  icon.addEventListener('mouseleave', function() {
    const infoboxId = icon.dataset.infoboxId;
    const infobox = document.getElementById(infoboxId);

    infobox.classList.remove('show');
  });
    }

});