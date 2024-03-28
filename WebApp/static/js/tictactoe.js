const socket = io.connect('http://' + document.domain + ':' + location.port + '/tictactoe');

const pp_button = document.getElementById("previouspage-link");
const np_button = document.getElementById("nextpage-link");
const board = document.getElementById("board");

const cellSize = 33.33;
const imageSize = 20;

var player1 = null;
var player2 = null;
var player = null;
var player_asset = "cross";
var playing = false;
var human_turn = false;
var human_choice = null;

for (let i = 0; i < 3; i++) {
  for (let j = 0; j < 3; j++) {
      const image_attributes = {
      class: "board-item",
      "data-index": i * 3 + j + 1,
      x: `${(j * cellSize) + (cellSize - imageSize) / 2}%`,
      y: `${(i * cellSize) + (cellSize - imageSize) / 2}%`,
      width: `${imageSize}%`,
      height: `${imageSize}%`,
      //display: "None",
      visibility: "hidden",
    };

    ["cross-shadow", "cross", "round-shadow", "round"].forEach(file_name => {
        const image = document.createElementNS("http://www.w3.org/2000/svg", "image");
        Object.keys(image_attributes).forEach(key => image.setAttributeNS(null, key, image_attributes[key]));
        image.setAttribute("href", `../static/icons/${file_name}.svg`);
        image.setAttribute("id", `${file_name}${i * 3 + j + 1}`);
        //image.setAttribute("style", "z-index: 99;");
        board.appendChild(image);
    });

    const cell_attributes = {
        id: `cell${i * 3 + j + 1}`,
      class: "board-cell",
      "data-index": "free",
      x: `${(j * cellSize)}%`,
      y: `${(i * cellSize)}%`,
      width: `${cellSize}%`,
      height: `${cellSize}%`,
      fill: "black",
      opacity: "1",
    };

    const cell = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    Object.keys(cell_attributes).forEach(key => cell.setAttributeNS(null, key, cell_attributes[key]));
    board.appendChild(cell);

    const new_cell=document.getElementById(`cell${i * 3 + j + 1}`)
    new_cell.addEventListener("mouseenter", () => {
        if (human_turn && new_cell.getAttribute("data-index")==="free"){
            if (player_asset === "cross") {
                document.getElementById(`cross-shadow${i * 3 + j + 1}`).style.visibility = "visible";
            } else if (player_asset === "round") {
                document.getElementById(`round-shadow${i * 3 + j + 1}`).style.visibility = "visible";
            }
        }
    });

    new_cell.addEventListener("mouseleave", () => {
            document.getElementById(`cross-shadow${i * 3 + j + 1}`).style.visibility = "hidden";
            document.getElementById(`round-shadow${i * 3 + j + 1}`).style.visibility = "hidden";
    });
    new_cell.addEventListener("click", () => {
        if (human_turn && new_cell.getAttribute("data-index")==="free"){
           human_choice = new_cell.id[new_cell.id.length-1];
           human_turn = false;

        }
    });
  }
}

function init_board() {
    document.querySelectorAll(`.board-cell`).forEach(element => element.setAttribute("data-index", "free"));
    document.querySelectorAll(`.board-item`).forEach(element => element.style.visibility = "hidden");
}

function take_cell(n){
    if (player_asset === "cross") {
        document.getElementById(`cross${n}`).style.visibility = "visible";
        player_asset = "round";
    } else if (player_asset === "round") {
        document.getElementById(`round${n}`).style.visibility = "visible";
        player_asset = "cross";
    }
    document.getElementById(`cell${n}`).setAttribute("data-index", "taken");}

function play() {
    playing = true;
    player_asset = "cross";
    init_board();
    player1 = document.getElementById('player1-selection').value;
    player2 = document.getElementById('player2-selection').value;
    console.log(player1+player2);

    document.querySelectorAll("#board > image").forEach(image => {
        image.style.opacity = "1";
    });
    
    socket.emit('play', {p1:player1, p2:player2});
    player = player1;
    disableButtons()
    game_loop();
}

async function game_loop(){
    var iterations = 0;
    while (iterations < 9 && playing){
        if (player != "0") {
            await new Promise(resolve => setTimeout(resolve, 1000));
            socket.emit('move');

        } else {
            human_turn = true;
            while (human_turn) {await new Promise(resolve => setTimeout(resolve, 100));};
            socket.emit('move', human_choice);
        }
        if (player === player1){
            player = player2;
        } else {
            player = player1;
        }
        iterations++;
        socket.emit('is_winner');
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    if (playing){
        end_game();
    }

}

function end_game(win_cells=[]){
    document.querySelectorAll("#board > image").forEach(image => {
        if (!win_cells.includes(parseInt(image.id[image.id.length-1]))){
            image.style.opacity = "0.5";
        }
    });
    reableButtons()
}

document.addEventListener('DOMContentLoaded', function() {
    socket.on('moved', function(data) {
        take_cell(data);
    });
    socket.on('winner', function(data) {
        playing = false;
        end_game(data);
    });
});

function disableButtons() {
    document.querySelectorAll('button').forEach(function(bouton) {bouton.disabled = true;});
    document.querySelectorAll('.round-button').forEach(function(bouton) {bouton.classList.add('disabled_button')});

    pp_button.style.cursor = "not-allowed";
    pp_button.style.opacity = "0.2";
    pp_button.classList.add('disabled_button');

    np_button.style.cursor = "not-allowed";
    np_button.style.opacity = "0.2";
    np_button.classList.add('disabled_button');

    // document.querySelectorAll('a').forEach(function(link) {link.addEventListener("click", function(event) {event.preventDefault();});});
    Array.from(document.links).forEach(link => {link.onclick = function(){return false;};});

    document.querySelectorAll('select').forEach(function(selection) {selection.disabled = true; selection.style.cursor = "not-allowed";});
}

function reableButtons() {
    isTraining = false;
    document.querySelectorAll('button').forEach(function(bouton) {bouton.disabled = false;});
    document.querySelectorAll('.round-button').forEach(function(bouton) {bouton.classList.remove('disabled_button')});

    pp_button.style.cursor = "pointer";
    pp_button.style.opacity = "1";
    pp_button.classList.remove('disabled_button');

    np_button.style.cursor = "pointer";
    np_button.style.opacity = "1";
    np_button.classList.remove('disabled_button');

    // document.querySelectorAll('a').forEach(function(link) {link.removeEventListener("click", function(event) {event.preventDefault();});});
    Array.from(document.links).forEach(link => {link.onclick = null;});

    document.querySelectorAll('select').forEach(function(selection) {selection.disabled = false; selection.style.cursor = "pointer";});
}


document.querySelectorAll('.info-icon').forEach(function(icon) {
    if (icon.dataset.infoboxId != "null") {
        icon.addEventListener('click', function(event) {
    const infoboxId = icon.dataset.infoboxId;
    console.log(infoboxId)
    const infobox = document.getElementById(infoboxId);

    infobox.style.left = event.clientX + 'px';
    infobox.style.top = event.clientY + 'px';

    infobox.classList.add('show');
  });

  icon.addEventListener('mouseleave', function() {
    const infoboxId = icon.dataset.infoboxId;
    const infobox = document.getElementById(infoboxId);

    infobox.classList.remove('show');
  });
    }

});