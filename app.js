const canvas = document.getElementById('predict-canvas');
const ctx = canvas.getContext('2d');
ctx.lineWidth = 10;
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

let isDrawing = false;

canvas.onmousedown = () => {
    isDrawing = true;
    startDrawing(event);
};

canvas.onmouseup = () => {
    isDrawing = false;
    ctx.beginPath();
};

canvas.onmousemove = draw;

function startDrawing(event) {
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.fillRect(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop, 1, 1);
    ctx.stroke();
}

function draw(event) {
    if (!isDrawing) return;
    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function convertToTensor() {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    let tensorData = [];
    for (let y = 0; y < canvas.height; y++) {
        let row = [];
        for (let x = 0; x < canvas.width; x++) {
            let idx = (y * canvas.width + x) * 4;
            let pixel = [imageData[idx], imageData[idx+1], imageData[idx+2], imageData[idx+3]];
            row.push(pixel);
        }
        tensorData.push(row);
    }
    
    const tensor = tf.tensor4d([tensorData]);
    tensor.print();
    return tensor;
}

function cropImage(img, width=140){
    img = img.slice([0,0,3])
    var mask_x = tf.greater(img.sum(0), 0).reshape([-1])
    var mask_y = tf.greater(img.sum(1), 0).reshape([-1])
    var st = tf.stack([mask_x,mask_y])
    var v1 = tf.topk(st)
    var v2 = tf.topk(st.reverse())
    
    var [x1, y1] = v1.indices.dataSync()
    var [y2, x2] = v2.indices.dataSync()
    y2 = width-y2-1
    x2 = width-x2-1
    var crop_w = x2-x1
    var crop_h = y2-y1
    
    if (crop_w > crop_h) {
        y1 -= (crop_w - crop_h) / 2
        crop_h = crop_w
    }
    if (crop_h > crop_w) {
        x1 -= (crop_h - crop_w) / 2
        crop_w = crop_h
    }
    
    img = img.slice([y1,x1],[crop_h,crop_w ])
    img = img.pad([[6,6],[6,6],[0,0]])
    var resized = tf.image.resizeNearestNeighbor(img,[28, 28])
    
    for(let i = 0; i < 28*28; i++) {
        resized[i] = 255 - resized[i]
    }
    return resized
}

function sendTensorData() {
    var preview = $('#preview-canvas')[0]
    
    var img = tf.browser.fromPixels(canvas, 4)
    var resized = cropImage(img, canvas.width)    
    tf.browser.toPixels(resized, preview)    
    
    var x_data = tf.cast(resized.reshape([1, 28, 28, 1]), 'float32')    
    console.log("Got data")
    x_data.print();
    console.log("array", x_data.arraySync());
    const address = document.getElementById('fetchAddress').value;

    img_data = {
        "instances" : x_data.arraySync()
    }
    console.log(img_data);
    // https://shared.bonn-test-50-edgexr.us.app.cloud.edgexr.org:10000/v1/models/digits-recognizer:predict
    fetch(address, {
        method: 'POST',
        body: JSON.stringify(img_data) // sending the reshaped tensor as a JSON array
    })
    .then(response => response.json())
    .then(data => {
        console.log("Success");
        console.log(data);
        var preds = data.predictions[0];
        console.log("Array:", preds);
        var prediction = preds.indexOf(Math.max(...preds));   
        $('#prediction').text( 'Predicted: '+ prediction);
    })
    .catch(error => {
        console.error('Error:', error)
        $('#prediction').text( 'Error: '+ error);
    });
}
