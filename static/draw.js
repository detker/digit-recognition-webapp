// Variables initialization.
let canvas;
let context;
let clickX = new Array();
let clickY = new Array();
let clickDrag = new Array();
let paint = false;
let curColor = "#FF5733";


// Preparing canvas - basic operations.
function drawCanvas() {
    canvas = document.getElementById('canvas');
    context = document.getElementById('canvas').getContext("2d");

    $('#canvas').mousedown(function (e) {
        let mouseX = e.pageX - this.offsetLeft;
        let mouseY = e.pageY - this.offsetTop;

        paint = true;
        addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
        redraw();
    });

    $('#canvas').mousemove(function (e) {
        if (paint) {
            addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
            redraw();
        }
    });

    $('#canvas').mouseup(function (e) {
        paint = false;
    });
}

// Saves the click postition.
function addClick(x, y, dragging) {
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
}

// Clearing the canvas and redrawing.
function redraw() {
    context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    context.strokeStyle = curColor;
    context.lineJoin = "round";
    context.lineWidth = 12;

    for (let i = 0; i < clickX.length; i++) {
        context.beginPath();
        if (clickDrag[i] && i) {
            context.moveTo(clickX[i - 1], clickY[i - 1]);
        } else {
            context.moveTo(clickX[i] - 1, clickY[i]);
        }
        context.lineTo(clickX[i], clickY[i]);
        context.closePath();
        context.stroke();
    }
}

// Encoding user input into base64 string and adding it to hidden form tag so Flask can get it.
function save() {
    let image = new Image();
    let url = document.getElementById('url');
    image.id = "pic";
    image.src = canvas.toDataURL();
    url.value = image.src;
}
