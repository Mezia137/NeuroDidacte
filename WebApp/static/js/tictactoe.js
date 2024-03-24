

const board = document.getElementById("board");

const cellSize = 33.33;
const imageSize = 20;

var player = "cross";
var playing = false;

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
        if (new_cell.getAttribute("data-index")==="free"){
            if (player === "cross") {
                document.getElementById(`cross-shadow${i * 3 + j + 1}`).style.visibility = "visible";
            } else if (player === "round") {
                document.getElementById(`round-shadow${i * 3 + j + 1}`).style.visibility = "visible";
            }
        }
    });

    new_cell.addEventListener("mouseleave", () => {
            document.getElementById(`cross-shadow${i * 3 + j + 1}`).style.visibility = "hidden";
            document.getElementById(`round-shadow${i * 3 + j + 1}`).style.visibility = "hidden";
    });
    new_cell.addEventListener("click", () => {
        if (new_cell.getAttribute("data-index")==="free"){
            if (player === "cross") {
                document.getElementById(`cross${i * 3 + j + 1}`).style.visibility = "visible";
                player = "round";
            } else if (player === "round") {
                document.getElementById(`round${i * 3 + j + 1}`).style.visibility = "visible";
                player = "cross";
            }
            new_cell.setAttribute("data-index", "taken");
        }
    });
  }
}

function init_board() {
    document.querySelectorAll(`.board-cell`).forEach(element => element.setAttribute("data-index", "free"));
    document.querySelectorAll(`.board-item`).forEach(element => element.style.visibility = "hidden");
}

function play() {
    playing = true;
    init_board()
}